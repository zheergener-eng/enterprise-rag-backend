# Evaluation Dataset v2 摘要

本阶段产物：更真实的企业 IT 运维知识库（knowledge_base_v2）+ 更困难的评测集（dataset_v2.json）。目标是用它更真实地测试 Vector Recall / Rerank / Relevance Gate / Query Rewrite / Multi-turn RAG，并为后续 Threshold Calibration 提供依据。

## 1. Knowledge Base v2 文档清单

| 文件 | 主题 | 中文字数 |
| --- | --- | --- |
| 01_xinghe_operations.md | 星河项目总体运维规范 | ~1542 |
| 02_database_backup_policy.md | 数据库备份与恢复管理规范 | ~1254 |
| 03_redis_operations.md | Redis 部署与运维规范 | ~1243 |
| 04_server_operations.md | 服务器部署与日常运维规范 | ~1208 |
| 05_security_policy.md | 权限控制与安全审计规范 | ~1214 |
| 06_incident_response.md | 生产故障响应与升级规范 | ~1218 |

预计按 chunk_size=600 / overlap=100 切片后，约产生 22~28 个 chunks。

## 2. 核心事实（跨文档一致）

- 星河项目生产库：每日 2:30 全量备份，保留 14 天
- 天枢项目生产库：每日 1:00 全量备份，保留 30 天
- 测试环境库：每日 4:00 全量备份，保留 7 天
- 临时验证环境：按需备份，最多保留 3 天
- Redis：默认端口 6379，生产用「主从 + 哨兵」，RDB 快照保留 3 天，AOF 保留 3 天
- 服务器：星河生产在 A3 机房，测试在 B1 机房；变更窗口每周二、四 22:00–次日 02:00
- 监控阈值：CPU >85%、内存 >90%（>95% 评估重启）、磁盘 <20%（<10% 立即扩容）、Redis 内存 >75%
- 安全：审计日志保留 90 天；最小权限；密钥/密码不得硬编码
- 故障：P1/P2/P3 三级；备份失败需登记升级

## 3. Dataset v2 类别与题量

| 类别 | 题数 |
| --- | --- |
| direct_answerable | 15 |
| paraphrase_answerable | 15 |
| entity_disambiguation | 12 |
| multi_turn（9 组 / 20 轮） | 20 |
| hard_negative | 24 |
| irrelevant | 10 |
| **合计** | **96** |

## 4. 故意设计的相似 / 冲突信息

- 多个「保留周期」并存：星河 14 天 / 天枢 30 天 / 测试 7 天 / 临时 3 天 / Redis 快照 3 天 / 审计日志 90 天，且「备份/保留/文件/恢复/日志」等词在多个文档反复出现，制造向量召回干扰。
- 多个「备份时间」并存：星河 2:30 / 天枢 1:00 / 测试 4:00，测试实体区分。
- 多个「机房」并存：生产 A3 / 测试 B1。
- 多个「90 天 / 3 天」类数字并存，测试数据类型区分（审计日志 90 天 vs 数据库备份 14 天 vs Redis 快照 3 天）。
- 「Redis 持久化」与「数据库备份」概念在 doc01/doc02/doc03 交叉出现，明确二者不是同一概念。

## 5. 故意设计的缺失信息（hard negative 锚点）

以下信息在 6 份文档中均**未给出答案**，用于构造「主题相关但无答案」的 hard negative：

- 星河备份的加密算法 / 压缩算法 / 存储设备品牌 / 恢复耗时
- 备份失败的责任团队 / 审批人 / 重试次数
- Redis 是否跨可用区 / 快照加密 / 分片算法 / 维护负责人
- A3 机房详细地址 / 服务器采购厂商 / 序列号 / 保修期 / 台数
- 最高权限员工 / 安全负责人姓名 / P1 审批人
- 生产访问审批时长 / 账号锁定阈值 / 主从切换耗时

## 6. Schema 说明

dataset_v2.json 每条记录包含：`id / category / question / answerable / expected_answer / expected_document_ids / expected_chunk_ids / notes`；multi_turn 额外含 `session_case_id / turn / history / expected_rewritten_query`。

- `expected_chunk_ids` 目前统一为空，待实际 Chunking 后生成 chunk mapping 再回填。
- 优先保留 `expected_document_ids` + `expected_answer / target_fact` 作为 ground truth。
- v2 使用 6 类 category（direct_answerable / paraphrase_answerable / entity_disambiguation / multi_turn / hard_negative / irrelevant），与 v1 的 3 类（answerable / irrelevant / hard_negative）不同，需要新的 loader 与评测脚本，后续阶段接入。

## 7. 使用约定

本阶段只生成知识库与评测集，**未运行任何 Evaluation**、未导入 Redis、未改 threshold、未根据模型分数调整题目。评测集生成逻辑独立于 Redis score / rerank score / threshold。
