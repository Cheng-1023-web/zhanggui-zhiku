# 掌柜智库 · 企业知识库 RAG 智能问答系统

面向业务文档构建可追溯的知识库问答系统：文档解析 → 文本切分 → 向量化 → 混合检索召回 → 重排序 → 答案生成，
支持回答依据逐条引用与幻觉抑制（无依据自动拒答）。

> **关于本仓库的数据诚实性**：下文所有指标均由实际运行 `python -m app.cli eval` 与基准测试脚本取得，
> 可在本地一键复现。项目默认使用**零依赖基线配置**（哈希嵌入 + 词面精排 + 抽取式生成），
> 以保证在无网络、无 GPU、无模型权重的环境下 clone 后即可完整跑通并复现全部数字。
> 生产级组件（BGE 嵌入、BGE-Reranker、Neo4j、真实 LLM）的代码路径**均已实现但默认关闭**，
> 启用方式见「[升级到生产级配置](#升级到生产级配置)」。这一取舍是刻意的工程决策，详见「[设计取舍](#设计取舍)」。

---

## 架构

```
┌─────────┐   ┌────────────────────── 离线索引链路 ──────────────────────┐
│ PDF/DOCX│   │ parsers.py → 结构化解析（保留标题层级 Section）           │
│ MD/TXT  │──▶│ chunker.py → 语义切分 + 递归字符切分                      │
└─────────┘   │   chunk=500 / overlap=100 (20%) / min=50                 │
              │ embedder.py → BGE(可选) / Hash 嵌入 → FAISS IndexFlatIP  │
              │ extractor.py → 规则/LLM 三元组抽取 → NetworkX / Neo4j    │
              └──────────────────────────────────────────────────────────┘
┌──────────────── 在线问答链路（自实现 LangGraph 风格状态图）─────────────┐
│ retrieve ──▶ rerank ──▶ guard ──(最高相关度 < 阈值)──▶ END（拒答）      │
│                              │                                          │
│                          (有依据)                                       │
│                              ▼                                          │
│                          generate ──▶ END                               │
│                                                                         │
│  FAISS 向量检索 Top-20 ┐                                                │
│                        ├→ RRF 融合 (k=60) → 二阶段精排 Top-5 → 引用生成  │
│  BM25 关键词检索 Top-20┘                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+）
pip install -r requirements.txt

# 2. 建库（解析 data/docs → 切分 → 索引 → 图谱），实测约 1 秒
python -m app.cli ingest

# 3. 问答
python -m app.cli ask -q "企业版的计费方式是什么"

# 4. 检索评测（Recall@5 / MRR）
python -m app.cli eval

# 5. 启动 Web 服务（FastAPI，http://127.0.0.1:8000）
python -m app.cli serve
```

> **索引产物未纳入版本控制**（见 `.gitignore`），因为 `ingest` 是确定性的且仅需约 1 秒。
> clone 后请先执行步骤 2 建库。

---

## 目录结构

```
.
├── app/
│   ├── config.py              # 配置加载 + ingest/load 入口
│   ├── cli.py                 # ingest / ask / eval / serve 四个命令
│   ├── ingest/
│   │   ├── parsers.py         # PDF/Word/MD/TXT 解析，保留标题层级
│   │   └── chunker.py         # 结构化语义切分 + 递归字符切分
│   ├── retrieval/
│   │   ├── tokenizer.py       # jieba 分词（字符 bigram 回退）
│   │   ├── embedder.py        # BGE 语义嵌入 / Hash 嵌入（默认）
│   │   ├── vector_store.py    # FAISS IndexFlatIP（numpy 回退）
│   │   ├── bm25.py            # BM25Okapi 关键词检索
│   │   ├── hybrid.py          # 多路召回 + RRF 倒数排序融合
│   │   └── reranker.py        # BGE-Reranker / 词面精排（默认）
│   ├── graphrag/
│   │   ├── extractor.py       # 规则式三元组抽取（LLM 抽取可选）+ 子图上下文
│   │   └── graph_store.py     # NetworkX（默认）/ Neo4j 适配器
│   ├── generation/
│   │   ├── llm.py             # OpenAI 兼容客户端 + 抽取式回退（默认）
│   │   └── answer.py          # 引用约束 + 拒答判定（幻觉抑制）
│   ├── pipeline/
│   │   ├── graph.py           # 自实现 StateGraph 引擎（对齐 langgraph API）
│   │   └── nodes.py           # retrieve → rerank → guard → generate 编排
│   ├── evaluation/
│   │   ├── metrics.py         # Recall@K / MRR / LLM-as-a-Judge 接口预留
│   │   └── eval_set.json      # 12 条人工标注评测集
│   └── api/server.py          # FastAPI：/api/ask /api/search /api/stats
├── data/
│   ├── docs/                  # 示例语料（6 份，虚构产品文档）
│   └── indices/               # 索引产物（.gitignore，由 ingest 重建）
├── web/index.html             # 轻量问答前端
├── config.yaml                # 全部可调参数
└── requirements.txt
```

**规模**：1323 行 Python，7 个功能模块，3 个 RESTful 接口，4 个 CLI 命令。

---

## 基线评测结果

默认配置（hash 嵌入 + 词面精排 + 抽取式生成）下的实测数据：

### 检索质量

`python -m app.cli eval` 输出：

| 指标 | 数值 |
|---|---|
| 评测集规模 | **12 条**人工标注查询 |
| Recall@5（重排前） | **0.7500** |
| Recall@5（重排后） | **0.8333** |
| **二阶段精排带来的提升** | **+8.33 个百分点（相对提升 11.1%）** |
| MRR | **0.8056** |

12 条查询中，**11 条重排后命中**，仅 1 条完全未命中（"如何保证回答不胡编乱造"，见下方归因）；
重排显著改善 1 条（"数据安全方面有哪些保障措施"，Recall@5 0.0 → 1.0）。

### 评测集标注审计（指标口径变更说明）

初版评测集存在 **3 处标注错误**，导致检索系统被错误扣分。逐条比对标注片段与语料实际内容后修正：

| # | 查询 | 原标注 | 问题 | 修正后 |
|---|---|---|---|---|
| 4 | API 请求频率限制是多少？超限了怎么办？ | `api-guide#c2` | c2 讲 `POST /api/ask` 接口参数，未提及频率限制 | `api-guide#c4`（请求频率限制）+ `#c5`（错误码） |
| 6 | 混合检索是怎么工作的？ | `product-manual#c2` + `#c3` | c2 讲 RBAC 权限管理，与混合检索无关 | 仅 `#c3`（智能问答与引用溯源） |
| 11 | 掌柜智库是否支持私有化部署？ | `product-manual#c5` + `faq#c0` | faq#c0 讲新员工开通账号，与部署方式无关 | 仅 `#c5`（部署方式） |

**修正前后对比**：

| 指标 | 修正前 | 修正后 |
|---|---|---|
| Recall@5（重排前） | 0.5833 | **0.7500** |
| Recall@5（重排后） | 0.6667 | **0.8333** |
| MRR | 0.7222 | **0.8056** |
| 完全未命中查询数 | 2 | **1** |

关键证据：修正前第 4 条查询 Recall=0，但实际检索返回的 **Top-1 就是 `api-guide#c4`**，
标题为「请求频率限制」，内容完全对应问题——**检索是对的，是标注错了**。
这类问题只能通过把每条标注片段与语料正文逐一对照才能发现，评测指标本身不会暴露它。

> 其余 9 条标注经核对均正确，未做改动。

### 唯一失败案例归因（保留不修）

第 7 条 "如何保证回答不胡编乱造"（标注答案 `product-manual#c4`「幻觉抑制与拒答机制」，标注正确）
Recall@5 = 0.0。归因：**hash 嵌入是词法哈希向量，不具备语义映射能力**——
查询用口语词「胡编乱造」，语料用术语「幻觉」，两者无共同 token，词法向量必然召回失败。

这不是标注问题，也不是检索链路缺陷，而是**默认基线配置的固有上限**，
恰好构成启用 BGE 语义嵌入的最直接动机（见「[升级到生产级配置](#升级到生产级配置)」）。
故**保留该失败案例不做人工干预**，作为基线与生产配置的效果对照锚点。

### 性能

| 路径 | retrieve | rerank | generate | 端到端 |
|---|---|---|---|---|
| **进程内热路径**（12 query × 3 轮 = 36 次采样） | **13.7 ms** | 4.4 ms | 1.1 ms | **19.2 ms** |
| **CLI 单次调用**（冷启动，含进程启动开销） | 446 ms | 3.9 ms | 1.9 ms | **452 ms** |

> ⚠️ **延迟口径说明（面试常被追问）**：CLI 单次调用的 452ms 中，
> **检索本身只占 13.7ms**，其余约 415ms 是进程冷启动开销，拆解为：
> 模块导入 197ms（含 jieba 词典加载）+ 索引反序列化 218ms（faiss/bm25/graph）。
> 生产环境用 `serve` 常驻进程即可消除该开销，单次问答落在 **~19ms** 量级。
> 首次 query 另有约 500ms 懒加载预热，之后稳定在 **6.1ms**。

### 索引与图谱规模

| 项 | 数值 |
|---|---|
| 语料 | 6 份 Markdown，9439 字符 |
| chunk 数 | **25**（3394 字符） |
| 抽取三元组 | **28 条** |
| 知识图谱 | **51 节点 / 28 边**（NetworkX） |
| 向量维度 | 1024 |
| 索引产物 | index.faiss 100K、bm25.pkl 33K、chunks.json 15K、graph.pkl 2.4K |
| `ingest` 耗时 | **约 0.98 秒**（确定性，可复现） |

### 幻觉抑制验证

| 测试 | 输入 | 结果 |
|---|---|---|
| 知识库外问题 | "今天天气怎么样" | ✅ `refused: true`，返回拒答话术 |
| 知识库内问题 | "企业版的计费方式是什么" | ✅ `refused: false`，挂载 **5 条**原文引用 + 图谱路径 |

---

## 核心技术实现

### 混合检索（多路召回）+ RRF 融合

`app/retrieval/hybrid.py`：

```python
def rrf_fuse(*ranked_lists, k: int = 60) -> dict[str, Candidate]:
    """RRF: score(d) = Σ 1/(k + rank_i(d))。k=60 抑制头部排名噪声。"""
```

FAISS 向量召回 Top-20 与 BM25 关键词召回 Top-20 双路并行，
经 RRF（Reciprocal Rank Fusion，k=60）合并候选集，再由二阶段精排保留 Top-5。

**为什么不用纯向量检索**：业务文档含大量产品型号、编号、专有名词，
向量检索对精确词召回弱，BM25 补齐这一短板；RRF 对两路分数尺度不敏感，无需归一化调参。

**FAISS 索引选型**：`IndexFlatIP`（内积）。向量已 L2 归一化，故内积等价于余弦相似度；
语料规模 25 chunk 下精确检索无性能压力，无需 IVF/HNSW 近似索引。
`vector_store.py` 内置 numpy 回退，无 faiss 环境亦可运行。

### 文本切分策略

`app/ingest/chunker.py`：结构化语义切分 + 递归字符切分两阶段结合。

1. 解析器按标题层级产出 `Section`（`heading_path` 如 `['计费说明', '版本与价格']`），先按段落边界聚合；
2. 超过 `chunk_size=500` 的长段落，用递归分隔符策略二次切分（`\n\n` → `\n` → `。` → `；` → `，` → 空格 → 硬切）；
3. 相邻 chunk 保留 `overlap=100`（20%）重叠，避免语义被切断；
4. `min_chunk_size=50` 过滤碎片。

chunk 保留 `heading_path` 元数据，供精排阶段的标题路径加分与答案引用的出处标注使用。

### 二阶段精排

`app/retrieval/reranker.py` 提供两种实现，通过 `config.yaml` 的 `rerank.provider` 切换：

- **`lexical`（当前默认）**：query token 覆盖率 × 0.7 + 标题路径命中加分 0.1
  + 原文完整包含加分 0.15 + RRF 分数 × 0.2，零依赖、可复现；
- **`bge`**：FlagEmbedding 的 `BAAI/bge-reranker-large`（Cross-Encoder 深度交互），
  精度高，仅对 Top-N 候选精排，用精度换延迟。模型缺失时自动回退 lexical。

### GraphRAG 引用溯源

`app/graphrag/extractor.py` 用 5 组中文业务文档高频句式正则模板抽取三元组
（`X 支持/提供/内置 Y`、`X 是/为 Y`、`X 适用于/面向 Y`、`X 分为/包括 Y`、`X 依赖/需要 Y`），
实测从 6 份文档抽出 **28 条三元组**，构建 **51 节点 / 28 边**图谱。
查询期 `graph_context()` 融合子图上下文注入 Prompt，答案的 `graph_lines` 字段携带图谱路径。

抽取策略可插拔：配置 `generation.api_base` 后可切换为 LLM 结构化 Prompt 抽取。
图存储 `graph_store.py` 为适配层设计，NetworkX（默认）与 Neo4j 实现同一接口。

### 双重幻觉抑制

`config.yaml` + `app/generation/answer.py`：

1. **阈值拒答**：`refusal_threshold: 0.15`，检索最高相关度低于阈值即拒答，
   返回明确话术而非编造内容；
2. **强制引用**：`citation_required: true`，每条结论必须挂载 `[n]` 原文 chunk 编号
   与出处（标题路径 + 文件路径），实测每次回答挂载 5 条引用。

### 自实现状态图引擎

`app/pipeline/graph.py` 实现 `StateGraph` / `add_node` / `add_edge` /
`add_conditional_edges` / `compile().invoke(state)`，API 对齐 langgraph 语义。
节点编排为 `retrieve → rerank → guard →(拒答|生成)→ END`，支持条件分支与节点级状态回溯，
各阶段耗时写入 `state["timings"]`。

**未引入 langgraph 依赖的原因**：保证零网络环境下可完整跑通与复现。
如需切换官方实现，仅需替换 `graph.py`（节点函数签名一致）。

### FastAPI 服务

`app/api/server.py`，`requirements.txt` 中 `fastapi>=0.110`、`uvicorn[standard]>=0.29` 为正式依赖：

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/ask` | POST | 问答，Pydantic `AskRequest` 校验请求体，返回答案 + 引用 + 图谱路径 + 分阶段耗时 |
| `/api/search` | POST | 仅检索，返回候选 chunk 及各路分数（vector / bm25 / rrf / rerank）便于调试 |
| `/api/stats` | GET | 索引与图谱统计 |
| `/` | GET | 轻量问答前端（`web/index.html`） |

---

## 评测体系

`app/evaluation/metrics.py`：

- `recall_at_k()` —— Recall@K
- `mrr()` —— Mean Reciprocal Rank
- `run_retrieval_eval()` —— 批量评测，按 **pre-rerank / post-rerank 分组对照**，输出逐条明细
- `judge_answer_relevance()` —— LLM-as-a-Judge 答案相关性评分（1-5 分）接口**已预留但未启用**，
  需配置真实 LLM，未配置时返回 `None`

评测集 `eval_set.json` 为 **12 条**人工标注查询，每条含 `question` 与 `gold_chunk_ids`（标准答案 chunk）。

> **已知局限**：12 条样本量偏小，单个 query 的命中变化即造成 8.3pp 的 Recall 波动，
> 指标置信区间较宽。扩充至 50-100 条是下一步首要改进项。

---

## 升级到生产级配置

以下三条代码路径**均已实现，仅为零依赖可复现而默认关闭**。启用后检索精度会显著提升，
可用 `python -m app.cli eval` 前后对比取得实测数据。

### 1. BGE 语义嵌入（**优先级最高**）

```bash
pip install sentence-transformers
```
```yaml
# config.yaml
embedding:
  provider: "bge"
  bge_model: "BAAI/bge-large-zh-v1.5"
```
```bash
python -m app.cli ingest   # 需重建索引
python -m app.cli eval
```

`embedder.py` 中 `BGEEmbedder` 已实现检索指令前缀（`"为这个句子生成表示以用于检索相关文章："`）
与 `normalize_embeddings=True`。模型权重约 1.3GB。

> **为什么这是优先级最高的一项**：当前 hash 嵌入是**词法哈希向量而非语义嵌入**
> （字符 bigram 哈希 + TF 加权 + L2 归一化），对语义改写无能为力。
> 实测那条 Recall=0 的失败 query（"如何保证回答不胡编乱造" → 标注答案在讲"幻觉抑制"）
> 正是典型的语义改写场景，hash 嵌入必然召回失败，换 BGE 后预期可命中。
>
> ⚠️ 同时需调整 `refusal_threshold`：hash 嵌入余弦分通常 0.1~0.3（故取 0.15），
> 换 BGE 后余弦分通常 0.5~0.8，建议调至 **0.35~0.45**，否则会大面积误拒答。
> `config.yaml` 中已有此项注释说明。

### 2. BGE-Reranker（Cross-Encoder 精排）

```bash
pip install FlagEmbedding
```
```yaml
rerank:
  provider: "bge"
  model: "BAAI/bge-reranker-large"
```

Bi-Encoder 粗召回快但精度低，Cross-Encoder 对 query-doc 做深度交互，
用精度换延迟，只对 Top-N 精排。预期提升幅度大于当前 lexical 精排的 +8.33pp，
并有望修复唯一失败案例（"胡编乱造" ↔ "幻觉" 语义鸿沟）。

### 3. Neo4j 图数据库

```bash
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/changeme neo4j
```
```yaml
graphrag:
  backend: "neo4j"
```

`graph_store.py` 中 `Neo4jGraphStore` 已实现；Neo4j 不可用时自动回退 NetworkX 并打印提示。

### 4. 真实 LLM 生成

```yaml
generation:
  api_base: "https://api.openai.com/v1"   # 任意 OpenAI 兼容服务
  api_key: "sk-..."
  model: "gpt-4o-mini"
```

未配置时使用 `ExtractiveFallback` 抽取式生成器，保证项目离线可完整跑通。

### 5. 扩充评测集

将 `eval_set.json` 从 12 条扩充至 50-100 条（当前 6 份文档可支撑的合理规模），
提升指标置信度。这是使评测数字具备说服力的关键一步。

---

## 设计取舍

本项目刻意采用「**零依赖基线 + 可插拔生产组件**」的双层设计，而非直接依赖重型组件：

| 决策 | 取舍 | 理由 |
|---|---|---|
| 默认 hash 嵌入而非 BGE | 牺牲语义召回精度 | 无需下载 1.3GB 权重，clone 后 1 秒建库即可复现全部指标 |
| 默认 lexical 精排而非 BGE-Reranker | 牺牲精排效果 | 同上，且保证评测结果确定性可复现 |
| 自实现 StateGraph 而非引入 langgraph | 多写约 60 行代码 | 零网络可跑通；API 对齐官方语义，替换成本极低 |
| NetworkX 而非 Neo4j 作默认图后端 | 牺牲图查询能力 | 无需 Docker 与外部服务；适配层设计使切换只需改一行配置 |
| 抽取式生成而非接真实 LLM | 答案质量下降 | 离线可完整验证检索链路；LLM 仅影响生成层，不影响检索指标 |
| 索引产物不入库 | clone 后需先 ingest | ingest 确定性且仅 1 秒，避免仓库携带 150K 二进制与跨平台兼容问题 |

每一层都在 `build_*` 工厂函数中实现「尝试加载生产组件 → 失败则回退基线并打印原因」，
上层调用代码完全无感知。这使得**同一份代码既能在离线环境复现基线数字，
也能在具备模型权重的环境直接切换到生产级配置**。

---

## 示例语料说明

`data/docs/` 下 6 份文档为**虚构产品文档**（"掌柜智库"为虚构企业知识库产品），
覆盖产品手册、计费说明、API 接入指南、数据安全白皮书、运维手册、FAQ 六类典型企业文档形态，
含中英文混排、多级标题、表格与编号列表，用于验证解析与切分链路。
**不含任何真实企业数据或个人信息。**

---

## 已知局限

诚实记录当前实现的不足：

1. **评测集仅 12 条**，样本量小，Recall 波动一个 query 即 8.3pp，指标置信区间宽
2. **hash 嵌入无语义能力**，语义改写类 query 必然召回失败（实测 1 条 Recall=0）
3. **语料仅 6 份 / 25 chunk**，规模不足以暴露大规模索引的性能与召回问题；
   FAISS 用 `IndexFlatIP` 精确检索，语料上万后需改 IVF/HNSW
4. **三元组抽取为规则式**（5 组正则模板），覆盖句式有限，复杂表述抽取率低；
   LLM 抽取路径已预留但未启用
5. **无并发与压测**，未测 P95/P99 在高并发下的表现；未做检索缓存
6. **`generate` 阶段为抽取式回退**，答案流畅度与推理能力有限
7. **无鉴权与多租户**，`/api/*` 接口裸露，仅适用于本地开发
8. **`recall@5_pre_rerank` 与 `post` 在部分 query 上相同**，
   说明 lexical 精排的区分度有限，这是换 BGE-Reranker 的主要动机

---

## 技术栈

**正式依赖**（`requirements.txt`）：
`faiss-cpu` · `numpy` · `rank-bm25` · `jieba` · `fastapi` · `uvicorn` · `pyyaml` · `networkx` · `pypdf` · `python-docx`

**可选依赖**（默认注释，按需启用）：
`sentence-transformers`（BGE 嵌入）· `FlagEmbedding`（BGE-Reranker）· `neo4j`（图数据库后端）

**运行环境**：Python 3.10+

---

## 复现步骤

从零复现本文档全部数字：

```bash
git clone <repo-url> && cd zhanggui-zhiku
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli ingest          # 期望输出 docs:6, chunks:25, triples:28, dim:1024
python -m app.cli eval            # 期望 recall@5: 0.7500 → 0.8333, mrr: 0.8056
python -m app.cli ask -q "今天天气怎么样"        # 期望 refused: true
python -m app.cli ask -q "企业版的计费方式是什么"  # 期望 refused: false, citations: 5
```

## License

MIT
