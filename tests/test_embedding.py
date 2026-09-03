"""Embedding 服务单元测试。

包含两类：
- 基于假模型的逻辑测试（快、不依赖网络/真实模型）
- 基于真实 bge-small-zh-v1.5 的语义相似度测试（首次需联网下载；real_embedder 由 conftest.py 提供）
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.embeddings import Embedder


# ---- 基于假模型的测试（不依赖真实模型 / 网络）----

def _make_fake_embedder(monkeypatch, dim: int = 8):
    """用假模型替换 SentenceTransformer，返回 (Embedder, 构造计数)。"""
    import sentence_transformers as st

    state = {"constructions": 0}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            state["constructions"] += 1

        def encode(self, texts, normalize_embeddings=False):
            texts = texts if isinstance(texts, list) else [texts]
            return np.zeros((len(texts), dim))

    monkeypatch.setattr(st, "SentenceTransformer", FakeModel)
    return Embedder("fake-model"), state


def test_embed_single_document(monkeypatch):
    """单个文本应生成一个向量。"""
    embedder, _ = _make_fake_embedder(monkeypatch)
    vecs = embedder.embed_documents(["这是一段文本。"])
    assert len(vecs) == 1
    assert isinstance(vecs[0], list)


def test_embed_multiple_chunks(monkeypatch):
    """多个 chunks 应返回相同数量的向量。"""
    embedder, _ = _make_fake_embedder(monkeypatch)
    chunks = ["第一段", "第二段", "第三段"]
    vecs = embedder.embed_documents(chunks)
    assert len(vecs) == len(chunks)


def test_embed_dimension_consistent(monkeypatch):
    """所有向量维度应一致。"""
    embedder, _ = _make_fake_embedder(monkeypatch, dim=8)
    vecs = embedder.embed_documents(["a", "b", "c"])
    assert {len(v) for v in vecs} == {8}


def test_model_initialized_once(monkeypatch):
    """多次 embed 调用只应初始化一次模型。"""
    embedder, state = _make_fake_embedder(monkeypatch, dim=8)
    embedder.embed_documents(["a"])
    embedder.embed_documents(["b", "c"])
    embedder.embed_query("问题")
    assert state["constructions"] == 1


# ---- 基于真实模型的测试（real_embedder fixture 来自 conftest.py）----

def _cosine(a: list[float], b: list[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_embed_dimension_matches_config(real_embedder):
    """真实模型输出维度应与配置 embedding_dim 一致（512）。"""
    from app.config import settings

    vec = real_embedder.embed_query("测试问题")
    assert len(vec) == settings.embedding_dim


def test_semantic_similarity_rank(real_embedder):
    """语义相似文本的相似度应高于无关文本。"""
    question = "如何安装 Python 依赖？"
    similar = "使用 pip install 命令可以安装 Python 包。"
    unrelated = "苹果是一种常见的水果。"

    qv = real_embedder.embed_query(question)
    sv = real_embedder.embed_documents([similar, unrelated])

    sim_similar = _cosine(qv, sv[0])
    sim_unrelated = _cosine(qv, sv[1])

    assert sim_similar > sim_unrelated
