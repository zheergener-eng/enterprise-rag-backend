# Evaluation v2 Report（Knowledge Base v2）

## 概览
- dataset_size: 96
- categories: {"direct_answerable": 15, "paraphrase_answerable": 15, "entity_disambiguation": 12, "multi_turn": 20, "hard_negative": 24, "irrelevant": 10}
- recall_top_n: 10
- rerank_top_k: 3
- gate_threshold: 0.5

## Chunk Manifest
- total_documents: 6
- total_chunks: 88
- per_document_chunk_count: {"01_xinghe_operations.md": 12, "02_database_backup_policy.md": 13, "03_redis_operations.md": 16, "04_server_operations.md": 15, "05_security_policy.md": 17, "06_incident_response.md": 15}
- avg_chunk_length: 148.98
- min_chunk_length: 52
- max_chunk_length: 413

## 数据完整性检查
- passed: True
- answerable_unmapped: []
- forbidden_signal_leaks: []

## Vector Retrieval（Recall 阶段）
| scope | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR |
|---|---|---|---|---|---|
| overall | 0.7903 | 0.9355 | 0.9355 | 0.8844 | 0.8575 |
| direct_answerable | 0.8 | 0.9333 | 0.9333 | 0.8778 | 0.8667 |
| paraphrase_answerable | 0.8 | 0.9333 | 0.9333 | 0.9111 | 0.8556 |
| entity_disambiguation | 0.8333 | 1.0 | 1.0 | 0.9167 | 0.9167 |
| multi_turn | 0.75 | 0.9 | 0.9 | 0.85 | 0.8167 |

## Rerank（Before → After）
- MRR before=0.8575  after=0.879
- Hit@1 before=0.7903  after=0.8387
- improved=5  unchanged=50  degraded=3  recall_miss=4

### Top improved examples
| question | correct_chunk | recall_rank | rerank_rank | vec_sim | rerank_score |
|---|---|---|---|---|---|
| 服务器的磁盘空间剩多少时需要立即清理或扩容？ | 04_server_operations.md:6 | 3 | 1 | 0.6221 | 0.9985 |
| 临时验证环境的备份文件最多保留几天？ | 02_database_backup_policy.md:3 | 2 | 1 | 0.5911 | 0.9964 |
| 临时验证环境的备份只保留多长时间？ | 02_database_backup_policy.md:3 | 2 | 1 | 0.5316 | 0.9871 |
| 哪个环境的备份文件最多只保留 3 天？ | 02_database_backup_policy.md:3 | 2 | 1 | 0.5724 | 0.9682 |
| Redis 的持久化和数据库备份是同一个概念吗？ | 03_redis_operations.md:4 | 2 | 1 | 0.7621 | 0.9995 |

### Top degraded examples
| question | correct_chunk | recall_rank | rerank_rank | vec_sim | rerank_score |
|---|---|---|---|---|---|
| 服务器内存使用率超过多少触发告警？ | 04_server_operations.md:6 | 1 | 2 | 0.6028 | 0.9653 |
| 测试环境的机器部署在哪个机房？ | 04_server_operations.md:2 | 1 | 2 | 0.5931 | 0.8525 |
| 它和数据库备份一样吗？ | 03_redis_operations.md:5 | 3 | None | 0.7152 | None |

## Relevance Gate
- Accuracy=0.9167  Precision=0.8857  Recall=1.0  F1=0.9394
- FPR=0.2353  FNR=0.0  (TP=62 FP=8 TN=26 FN=0)
- answerable: {"count": 62, "tp": 62, "fn": 0, "pass_rate": 1.0}
- hard_negative: {"count": 24, "tn": 16, "fp": 8, "rejection_rate": 0.6667}
- irrelevant: {"count": 10, "tn": 10, "fp": 0, "rejection_rate": 1.0}

## Score Distribution（top-1 rerank score）
| class | count | min | max | mean | median | p25 | p75 | p90 | p95 |
|---|---|---|---|---|---|---|---|---|---|
| answerable | 62 | 0.8056 | 0.9999 | 0.9836 | 0.9968 | 0.9797 | 0.9991 | 0.9995 | 0.9997 |
| hard_negative | 24 | 0.0142 | 0.9759 | 0.3753 | 0.1987 | 0.0812 | 0.7159 | 0.8482 | 0.9207 |
| irrelevant | 10 | 0.0 | 0.0073 | 0.0015 | 0.0001 | 0.0 | 0.0017 | 0.0059 | 0.0073 |
- answerable_range: [0.8056, 0.9999]
- hard_negative_range: [0.0142, 0.9759]
- irrelevant_range: [0.0, 0.0073]
- answerable↔hard_negative overlap: 0.1703（明显）
- answerable↔irrelevant overlap: 0.0（很小）

## Threshold Sweep
| threshold | TP | FP | TN | FN | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 62 | 34 | 0 | 0 | 0.6458 | 1.0 | 0.7848 | 1.0 | 0.0 |
| 0.1 | 62 | 16 | 18 | 0 | 0.7949 | 1.0 | 0.8857 | 0.4706 | 0.0 |
| 0.2 | 62 | 12 | 22 | 0 | 0.8378 | 1.0 | 0.9118 | 0.3529 | 0.0 |
| 0.3 | 62 | 11 | 23 | 0 | 0.8493 | 1.0 | 0.9185 | 0.3235 | 0.0 |
| 0.4 | 62 | 9 | 25 | 0 | 0.8732 | 1.0 | 0.9323 | 0.2647 | 0.0 |
| 0.5 | 62 | 8 | 26 | 0 | 0.8857 | 1.0 | 0.9394 | 0.2353 | 0.0 |
| 0.6 | 62 | 8 | 26 | 0 | 0.8857 | 1.0 | 0.9394 | 0.2353 | 0.0 |
| 0.7 | 62 | 7 | 27 | 0 | 0.8986 | 1.0 | 0.9466 | 0.2059 | 0.0 |
| 0.8 | 62 | 5 | 29 | 0 | 0.9254 | 1.0 | 0.9612 | 0.1471 | 0.0 |
| 0.9 | 61 | 2 | 32 | 1 | 0.9683 | 0.9839 | 0.976 | 0.0588 | 0.0161 |
| 0.95 | 57 | 1 | 33 | 5 | 0.9828 | 0.9194 | 0.95 | 0.0294 | 0.0806 |
| 0.98 | 46 | 0 | 34 | 16 | 1.0 | 0.7419 | 0.8519 | 0.0 | 0.2581 |

## Multi-turn / Query Rewrite
- turn_count=20  rewrite_used_llm=11  rewrite_fallback=0
- recalled_hit_rate=0.9  reranked_hit_rate=0.85
| id | turn | rewritten_query | expected_rewritten_query | recalled_hit | reranked_hit |
|---|---|---|---|---|---|
| m001_t1 | 1 | None | None | True | True |
| m001_t2 | 2 | 星河项目数据库的备份文件会保留多久？ | 星河项目数据库备份文件保留多久 | True | True |
| m002_t1 | 1 | None | None | True | True |
| m002_t2 | 2 | Redis RDB快照文件和数据库备份一样吗？ | Redis 快照保留时间与数据库备份保留时间是否相同 | True | False |
| m003_t1 | 1 | None | None | True | True |
| m003_t2 | 2 | 星河项目数据库备份保留多久？ | 星河项目数据库备份保留多久 | True | True |
| m004_t1 | 1 | None | None | True | True |
| m004_t2 | 2 | 测试环境数据库的备份文件保留几天？ | 测试环境数据库备份文件保留几天 | True | True |
| m005_t1 | 1 | None | None | True | True |
| m005_t2 | 2 | 测试服务器在哪个机房？ | 测试服务器在哪个机房 | True | True |
| m006_t1 | 1 | None | None | True | True |
| m006_t2 | 2 | Redis 哨兵模式的默认端口是多少？ | Redis 默认服务端口是多少 | True | True |
| m007_t1 | 1 | None | None | False | False |
| m007_t2 | 2 | 服务器内存使用率持续超过多少需要评估重启？ | 服务器内存使用率持续超过多少需评估重启 | False | False |
| m008_t1 | 1 | None | None | True | True |
| m008_t2 | 2 | 星河项目数据库的备份文件会保留多久？ | 星河项目数据库备份文件保留多久 | True | True |
| m008_t3 | 3 | 天枢项目的备份保留多久？ | 天枢项目数据库备份保留多久 | True | True |
| m009_t1 | 1 | None | None | True | True |
| m009_t2 | 2 | 测试环境数据库备份保留多久？ | 测试环境数据库备份保留多久 | True | True |
| m009_t3 | 3 | 临时验证环境的数据库备份保留多久？ | 临时验证环境备份保留多久 | True | True |

## Answer Check（轻量 deterministic substring check）
- passed=54  failed=1  skipped=0  compound_skipped=7  pass_rate=0.9818
- note: Answer Check 为轻量 deterministic substring check，不能等同于完整语义正确性评估。

## Latency
- avg_embed=0.0078s  p95_embed=0.01s
- avg_recall=0.0023s  p95_recall=0.0029s
- avg_rerank=0.7428s  p95_rerank=0.8801s
- avg_total_retrieval=0.7529s  avg_candidates=10

## Relevance Threshold Selection

Evaluation v2 显示，单一 CrossEncoder relevance score **无法完全分离**
可作答查询与「主题高度相关但知识库没有答案」的 hard negative。

在当前 benchmark 上的 threshold sweep：

| threshold | TP | FP | TN | FN | Precision | Recall | FPR |
|---|---|---|---|---|---|---|---|
| 0.5 | 62 | 8 | 26 | 0 | 0.886 | 1.0 | 0.235 |
| 0.8 | 62 | 5 | 29 | 0 | 0.925 | 1.0 | 0.147 |
| 0.9 | 61 | 2 | 32 | 1 | 0.968 | 0.984 | 0.059 |

因此选定 **0.8** 作为生产默认阈值（`RERANK_RELEVANCE_THRESHOLD`）：
- 相较 0.5，FP 从 8 降至 5、FPR 从约 23.5% 降至约 14.7%、Precision 从约 88.6% 提升到约 92.5%，且 FN 仍为 0、Recall 保持 100%；
- 升至 0.9 虽进一步压低 FP，但首次产生 FN=1、Recall 开始下降。

**Selected operating threshold: 0.8**

该值是**根据当前 Evaluation v2 benchmark 选择的工程折中阈值**，既不是理论最优值，也不普适。当知识库、Embedding 模型、Reranker 或真实业务 query 分布发生变化时，应重新校准。

### Known limitation（已知局限）

Evaluation v2 中 answerable 的最低 top-1 rerank score ≈ 0.8056，hard_negative 的最高 score ≈ 0.9759，两类存在 score overlap。这说明 CrossEncoder relevance score 更接近「语义相关程度」，并不等同于「该 chunk 中一定包含回答问题所需的充分证据」。因此单一 threshold 无法彻底消除「主题高度相关但知识库没有答案」的 false positive。这是当前系统的 known limitation，本阶段不做 Multi-signal Gate 等改造。
