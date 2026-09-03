"""离线 Evaluation Framework 包。

职责：量化评测 RAG 检索链路（Vector Recall / Rerank / Relevance Gate），
为后续 Threshold Calibration 提供依据。评测代码独立于 app/services 主业务代码，
并使用隔离的 Redis index / key prefix（见 knowledge.EVAL_INDEX_NAME）。
"""
