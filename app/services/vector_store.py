"""Redis Stack 向量存储封装。

使用 RediSearch 的 HNSW 向量索引持久化 chunk 及其向量，距离度量 COSINE。
提供「写入 / 读取 / KNN 检索」能力。

存储结构：每个 chunk 一个 Redis Hash，key 形如 `{prefix}{document_id}:{index}`，
字段含 document_id / chunk_id / index / text / title / embedding（float32 字节）。
RediSearch 通过 key 前缀自动索引匹配的 Hash。

KNN 检索（search）按 COSINE 距离升序返回结果；为便于下游理解，
每项同时给出 similarity = 1 - distance（余弦相似度，越大越相关）。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from redis import Redis
from redis.exceptions import ResponseError
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.config import settings

# Redis SCAN 的 match 参数使用 glob 语法，这些字符会被当作通配符。构建
# ``delete_document`` / ``get_document_chunks`` 的匹配模式时需转义，避免
# 用户可控的 document_id 中含 ``* ? [ ] \`` 时误匹配其它文档的 key。
_GLOB_SPECIAL_CHARS = ("\\", "*", "?", "[", "]")


def _escape_glob(text: str) -> str:
    """转义 Redis glob 特殊字符，使其按字面匹配（每个特殊字符前加 ``\\``）。"""
    out: list[str] = []
    for ch in text:
        if ch in _GLOB_SPECIAL_CHARS:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


class VectorDimensionError(ValueError):
    """写入或检索的向量维度与索引配置不符（应用层提前校验，避免拖到 Redis 才报错）。"""


class VectorStore:
    """基于 Redis Stack / RediSearch 的向量存储。"""

    def __init__(
        self,
        redis_url: str,
        index_name: str,
        dim: int,
        prefix: str | None = None,
    ) -> None:
        self.redis_url = redis_url
        self.index_name = index_name
        self.dim = dim
        # 前缀默认取 settings.chunk_prefix（生产默认 chunk:），测试 / Evaluation
        # 通过显式传入独立 prefix 实现 key 命名空间隔离。
        self.prefix = prefix if prefix is not None else settings.chunk_prefix
        # decode_responses=False：embedding 以 float32 字节存储，需保留二进制
        self._client = Redis.from_url(redis_url, decode_responses=False)

    # ---- 维度校验 ----

    def _check_vector_dim(self, vec, index: int | None = None) -> None:
        """校验向量维度与索引配置一致，不一致抛出 VectorDimensionError。"""
        if len(vec) != self.dim:
            where = f" at index {index}" if index is not None else ""
            raise VectorDimensionError(
                f"vector dimension mismatch{where}: expected {self.dim}, got {len(vec)}"
            )

    # ---- 索引管理 ----

    def index_exists(self) -> bool:
        """索引是否已存在。"""
        try:
            self._client.ft(self.index_name).info()
            return True
        except ResponseError:
            return False

    def ensure_index(self) -> None:
        """幂等创建向量索引（已存在则跳过，重复启动不报错）。"""
        if self.index_exists():
            return
        schema = (
            TagField("document_id"),
            TagField("chunk_id"),
            NumericField("index"),
            TextField("text"),
            TextField("title"),
            VectorField(
                "embedding",
                algorithm="HNSW",
                attributes={
                    "TYPE": "FLOAT32",
                    "DIM": self.dim,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
        self._client.ft(self.index_name).create_index(schema, definition=definition)

    def drop_index(self) -> None:
        """删除索引及其文档（测试清理用）。"""
        try:
            self._client.ft(self.index_name).dropindex(delete_documents=True)
        except ResponseError:
            pass

    # ---- 写入 ----

    def _chunk_key(self, document_id: str, index: int) -> str:
        return f"{self.prefix}{document_id}:{index}"

    def _scan_keys(self, pattern: str) -> list[bytes]:
        """扫描匹配 pattern 的所有 key。"""
        keys: list[bytes] = []
        cursor = 0
        while True:
            cursor, batch = self._client.scan(cursor, match=pattern, count=100)
            keys.extend(batch)
            if cursor == 0:
                break
        return keys

    def delete_document(self, document_id: str) -> int:
        """删除某 document_id 的所有 chunk（RediSearch 自动移除索引）。

        document_id 中的 glob 特殊字符（``* ? [ ] \\``）会被转义后再拼 SCAN
        匹配模式，避免误匹配其它文档的 chunk key。
        """
        keys = self._scan_keys(f"{self.prefix}{_escape_glob(document_id)}:*")
        if keys:
            self._client.delete(*keys)
        return len(keys)

    def add_document(
        self,
        document_id: str,
        chunks: list[str],
        vectors: list[list[float]],
        title: str = "",
        filename: str = "",
    ) -> int:
        """覆盖式写入一个文档的所有 chunk。

        策略：先删除该 document_id 的旧 chunk，再写入新 chunk，
        保证同一 document_id 重复导入不会产生重复数据。

        filename / title 作为文档元数据写入每个 chunk 的 Hash（不参与索引，
        仅用于溯源），避免外部调用方丢失原始文件名。
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        # 应用层提前校验向量维度，避免写入错误维度、直到 Redis 检索时才报 dimension mismatch。
        # 校验在 delete_document 之前完成，保证维度不符时不会先删旧数据造成半写状态。
        for i, vec in enumerate(vectors):
            self._check_vector_dim(vec, index=i)

        self.delete_document(document_id)

        pipe = self._client.pipeline(transaction=False)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            fields = {
                "document_id": document_id,
                "chunk_id": f"{document_id}:{i}",
                "index": i,
                "text": chunk,
                "title": title or "",
                "filename": filename or "",
                "embedding": np.asarray(vec, dtype=np.float32).tobytes(),
            }
            pipe.hset(self._chunk_key(document_id, i), mapping=fields)
        pipe.execute()
        return len(chunks)

    # ---- 读取（测试 / 验证用）----

    def get_document_chunks(self, document_id: str) -> list[dict]:
        """读取某文档的所有 chunk（按 index 排序）。"""
        keys = self._scan_keys(f"{self.prefix}{_escape_glob(document_id)}:*")
        results: list[dict] = []
        for key in keys:
            data = self._client.hgetall(key)
            results.append(
                {
                    "document_id": data[b"document_id"].decode("utf-8"),
                    "chunk_id": data[b"chunk_id"].decode("utf-8"),
                    "index": int(data[b"index"]),
                    "text": data[b"text"].decode("utf-8"),
                    "title": data[b"title"].decode("utf-8"),
                    "filename": data.get(b"filename", b"").decode("utf-8"),
                    "embedding": np.frombuffer(data[b"embedding"], dtype=np.float32),
                }
            )
        results.sort(key=lambda x: x["index"])
        return results

    # ---- 检索（KNN）----

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """KNN 检索：按 COSINE 距离返回最相近的 top_k 个 chunk。

        使用 RediSearch 的 `*=>[KNN k @embedding $vec AS distance]` 语法，
        按 distance 升序（即相关性从高到低）返回结果。

        Args:
            query_vector: 查询向量（与写入同维度，float32 列表）。
            top_k: 返回的最大结果数；大于实际 chunk 数时返回全部。

        Returns:
            结果列表，按 COSINE 距离升序。每项包含：
            - document_id / chunk_id / index / text / title：chunk 元数据
            - distance：COSINE 距离（范围 [0,2]，越小越相关）
            - similarity：余弦相似度 1 - distance（越大越相关）

        说明：RediSearch 的 COSINE 距离定义为 1 - cosine_similarity，
        对 L2 归一化向量（本模型输出已归一化）等价于 1 - 点积。
        因此 distance=0 表示完全一致，distance=2 表示完全相反。
        """
        self._check_vector_dim(query_vector)
        vec = np.asarray(query_vector, dtype=np.float32).tobytes()

        query = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS distance]")
            .return_fields("document_id", "chunk_id", "index", "text", "title", "distance")
            .sort_by("distance")
            .dialect(2)
            .paging(0, top_k)
        )
        res = self._client.ft(self.index_name).search(
            query, query_params={"vec": vec}
        )

        results: list[dict] = []
        for doc in res.docs:
            distance = float(doc.distance)
            results.append(
                {
                    "document_id": doc.document_id,
                    "chunk_id": doc.chunk_id,
                    "index": int(doc.index),
                    "text": doc.text,
                    "title": getattr(doc, "title", "") or "",
                    "distance": distance,
                    "similarity": 1.0 - distance,
                }
            )
        return results


@lru_cache
def get_vector_store() -> VectorStore:
    """返回进程级单例 VectorStore（生产默认 index / prefix）。"""
    return VectorStore(
        redis_url=settings.redis_url,
        index_name=settings.redis_index_name,
        dim=settings.embedding_dim,
        prefix=settings.chunk_prefix,
    )


# 便捷单例
vector_store = get_vector_store()
