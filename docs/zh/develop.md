# 开发文档

## 1. 架构总览

CiteVerifier 分为两个执行面：

1. **CLI 验证链路**（`verifier.py`）
   - 执行“DBLP/缓存/在线源/LLM 重解析”的策略链。
   - 输出验证结果与统计。
2. **Web 检索服务**（`web_app.py`）
   - 提供单条与批量标题检索。
   - 记录运行时统计到 `runtime.sqlite`。

配套模块：

- `unified_database.py`：验证缓存数据库（`scholar_results` / `search_results`）。
- `runtime_store.py`：Web 运行态统计与批次记录。
- `dblp_match.py`：本地 DBLP 检索函数（索引路径 + brute-force 回退）。

## 2. CLI 验证链路细节（含缓存）

### 2.1 策略链装配规则

`ScrapingDogVerifier._build_verification_chain()` 的真实行为：

- 默认（`--enable-online` 未开启）：
  1. `DblpVerificationStrategy`
  2. `CacheVerificationStrategy`
- 开启 `--enable-online` 后追加：
  3. `APIVerificationStrategy`
  4. `GoogleFallbackStrategy`
  5. `LLMReparseStrategy`

如果在线客户端导入失败（依赖缺失或配置问题），会自动回退到 DBLP+缓存模式。

### 2.2 CLI 缓存读写语义

缓存库为 `scholar_results.db`（`UnifiedDatabase`）：

- **读路径**：`CacheVerificationStrategy` 使用 `search_scholar_by_title(title)`。
  - SQL 为 `WHERE title LIKE '%{title}%' ORDER BY created_at DESC LIMIT 1`。
  - 命中后再走统一验证器，不是无条件直接判定 VALID。
- **写路径**：仅当最终验证为 `VALID` 且存在 `best_match` 时，写入 `scholar_results`。
- **去重约束**：唯一索引 `UNIQUE(title, authors, year)`；重复写入默认忽略。

这意味着 CLI 缓存是“结果缓存”，并非“请求缓存”，且按 title LIKE 检索时可能存在近似命中风险。

## 3. Web 检索链路与缓存

### 3.1 单条检索流程（`POST /api/search/title`）

1. 标题规范化（去首尾空白、压缩多空格）。
2. 检查 DBLP 文件存在，不存在则 `404`。
3. 检索策略：
   - 若检测到词索引，走 indexed 检索。
   - 否则回退 brute-force。
4. 若命中，补充 `publications` 元信息（`year/venue/pub_type`）。
5. 记录 runtime 计数与单次事件。

### 3.2 批量检索流程（`POST /api/search/title/batch`）

1. 标题列表先标准化 + 去重（`casefold` 级别）。
2. 空列表返回 `400`；超过 `MAX_BATCH_TITLES=200` 返回 `400`。
3. 请求内按 `for` 循环逐条处理（串行），每条写 `batch_items`。
4. 批次结束后写 `batch_runs` 汇总与计数器。

### 3.3 Web 内存缓存语义

`web_app.py` 有 `_brute_cache: dict[db_path -> all_titles]`：

- 仅用于 brute-force 场景。
- 首次加载由 `_brute_cache_lock` 保护，后续复用。
- 无 TTL、无大小上限、无主动淘汰。
- 进程重启后失效（纯内存）。

## 4. 并发模型与超限行为（等待 / 拒绝 / 降级）

| 场景 | 约束 | 超限行为 | 结果 |
|---|---|---|---|
| CLI 批量验证并发 | `asyncio.Semaphore(max_concurrent)`，默认 `--concurrent=10` | 超出并发时等待信号量 | 任务排队继续执行，不拒绝 |
| CLI `--concurrent` 参数 | 代码无硬上限 | 不自动拒绝 | 可能放大 IO 压力（由外部资源限流） |
| Web 批量条数 | 最多 `200` | 立即拒绝 | `400 Batch size exceeds limit` |
| Web `max_candidates` | `1..500000`（Pydantic） | 立即拒绝 | `422` |
| Web 首次 brute-cache 装载 | `_brute_cache_lock` 互斥 | 后续请求等待锁 | 等待完成后继续 |
| Web 批请求内单条检索 | 串行 `for` 循环 | 无“超限拒绝”，仅处理时间增长 | 请求耗时变长 |
| Web runtime SQLite 写冲突 | `sqlite timeout=30s` + `WAL` | 先等待锁 | 超时后异常（通常 `500`） |
| CLI 在线客户端不可用 | 导入或初始化失败 | 自动降级 | 仅 DBLP+缓存继续运行 |

关键回答：

- CLI 并发超出是**等待**（不是禁止）。
- Web 批量条数超出是**禁止并返回 400**。
- SQLite 锁冲突是**先等待，超时再失败**。

## 5. 参数约束与错误码细化

### 5.1 Web API

- `POST /api/search/title`
  - `title`：最短 1 字符；空值会 `422`。
  - `max_candidates`：`1..500000`；越界 `422`。
- `POST /api/search/title/batch`
  - `titles`：清洗去重后必须至少 1 条，否则 `400`。
  - 清洗后条数 > 200，返回 `400`。
- `GET /api/health`
  - DB 不存在时返回 `{"status":"error"}`（HTTP 200）。

### 5.2 CLI 关键参数

- `--dblp-db`：本地 DBLP sqlite 路径。
- `--dblp-threshold`：DBLP 标题相似度阈值（默认 `0.9`）。
- `--dblp-max-candidates`：DBLP 候选上限（默认 `100000`）。
- `--disable-dblp`：跳过 DBLP 预匹配。
- `--enable-online`：启用在线策略链。

## 6. Runtime 数据模型（Web）

`runtime_store.py` 表结构：

- `runtime_counters`
- `single_search_events`
- `batch_runs`
- `batch_items`
- `event_logs`

建议重点观测：

- `single_search_requests / single_search_errors`
- `batch_search_requests`
- `batch_search_items_total / batch_search_found_total`
- `batch_runs.status` 与 `error_message`

## 7. 开发建议（面向缓存与并发）

- 若要引入跨进程共享缓存，优先把 `_brute_cache` 升级为显式可控存储（带 TTL 与容量控制）。
- 若要提升 Web 批量吞吐，可在请求内增加有限并发，但需同步设计 DB 连接池与超时策略。
- 若要强化 CLI 在线稳定性，建议增加全局 LLM/API 限流器与退避重试策略。
- 对每个新接口必须明确“超限时是等待、拒绝还是降级”，并在文档中固定该语义。