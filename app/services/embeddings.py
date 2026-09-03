"""向量化（Embedding）封装。

基于 sentence-transformers，默认使用 BAAI/bge-small-zh-v1.5（512 维）。
- 文档与查询使用同一模型；
- 查询侧附加 bge 官方指令前缀以提升检索对齐效果；
- 模型懒加载，进程内仅初始化一次（避免每次请求重复加载）。
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings

# bge 系列模型的官方查询指令前缀（仅查询侧使用，文档侧不加）
_BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    """文本向量化封装（懒加载 + 进程内复用同一模型实例）。"""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None  # 懒加载：首次使用时才真正加载模型

    def _load_model(self):
        """加载模型（仅首次调用时执行，后续复用同一实例）。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        """将文档片段列表编码为向量列表。

        Args:
            chunks: 文档片段文本列表。

        Returns:
            与输入等长、同维度的向量列表（已 L2 归一化）。
        """
        model = self._load_model()
        embeddings = model.encode(chunks, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, question: str) -> list[float]:
        """将查询文本编码为向量（附加 bge 查询指令前缀）。"""
        model = self._load_model()
        query = f"{_BGE_QUERY_INSTRUCTION}{question}"
        embedding = model.encode(query, normalize_embeddings=True)
        return embedding.tolist()


@lru_cache
def get_embedder() -> Embedder:
    """返回进程级单例 Embedder（模型只初始化一次）。"""
    return Embedder(model_name=settings.embedding_model)


# 便捷单例：`from app.services.embeddings import embedder`
embedder = get_embedder()
