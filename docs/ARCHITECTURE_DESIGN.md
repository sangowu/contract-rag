# CUAD 合同助手 - 完整设计框架

> 文档版本: 1.1  
> 更新日期: 2025-01-26  
> 更新内容: 补充 PDF坐标溯源、表格处理策略、语义缓存、用户反馈闭环

---

## 一、系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           用户层 (Frontend)                                  │
│  Next.js 14+ / TypeScript / 流式响应 / PDF预览器 / 进度展示                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         编排层 (Orchestration)                               │
│  Next.js API Routes / BullMQ任务队列 / Next-Auth鉴权 / PostgreSQL元数据       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI服务层 (Python FastAPI)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  PDF解析引擎  │  │  RAG检索服务  │  │  LLM推理服务  │  │  评估服务     │     │
│  │  Marker/OCR  │  │  Embedding   │  │  vLLM/Qwen3  │  │  RAGAS/WandB │     │
│  └──────────────┘  │  Reranker    │  └──────────────┘  └──────────────┘     │
│                    │  ChromaDB    │                                         │
│                    └──────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           基础设施层 (Infrastructure)                         │
│  Docker / MinIO(S3) / Redis / PostgreSQL / GPU调度 / 监控(LangSmith)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、目录结构设计

```
cuad-assistant/
├── config/
│   ├── base.yaml              # 公共配置
│   ├── dev.yaml               # 开发环境配置
│   ├── test.yaml              # 测试环境配置 (10%数据采样)
│   └── prod.yaml              # 生产环境配置
│
├── src/
│   ├── core/                  # 核心抽象层
│   │   ├── __init__.py
│   │   ├── config.py          # 配置加载器 (支持 --mode 参数)
│   │   ├── exceptions.py      # 全局异常定义
│   │   └── interfaces.py      # 抽象接口定义
│   │
│   ├── pdf/                   # PDF解析模块 [新增]
│   │   ├── __init__.py
│   │   ├── parser.py          # PDF解析主逻辑
│   │   ├── ocr_engine.py      # OCR处理 (PaddleOCR)
│   │   ├── table_extractor.py # 表格提取
│   │   ├── layout_analyzer.py # 版面分析 (Marker)
│   │   ├── exceptions.py      # PDF相关异常
│   │   └── tests/
│   │       └── test_parser.py
│   │
│   ├── data/                  # 数据处理模块 [重构]
│   │   ├── __init__.py
│   │   ├── loader.py          # 数据加载 (解耦路径)
│   │   ├── chunker.py         # 分块策略
│   │   ├── gold_standard.py   # 金标准生成
│   │   └── tests/
│   │       └── test_loader.py
│   │
│   ├── rag/                   # RAG检索模块 [已有,需重构]
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   ├── retrieval.py
│   │   ├── reranker.py
│   │   └── tests/
│   │       └── test_retrieval.py
│   │
│   ├── inference/             # LLM推理模块 [已有]
│   │   ├── __init__.py
│   │   ├── llm_inference.py
│   │   └── tests/
│   │       └── test_inference.py
│   │
│   ├── evaluation/            # 评估模块 [新增]
│   │   ├── __init__.py
│   │   ├── ragas_adapter.py   # RAGAS格式转换
│   │   ├── metrics.py         # 传统指标 (F1, Recall, MRR)
│   │   ├── wandb_reporter.py  # WandB上报
│   │   ├── pipeline.py        # 评估管道
│   │   └── tests/
│   │       └── test_metrics.py
│   │
│   └── utils/                 # 工具函数 [已有]
│       ├── __init__.py
│       ├── prompting.py
│       ├── plot.py
│       └── model_loading.py
│
├── api/                       # FastAPI服务
│   ├── __init__.py
│   ├── main.py                # FastAPI入口
│   ├── routes/
│   │   ├── pdf.py             # PDF上传/解析接口
│   │   ├── query.py           # 问答接口
│   │   └── evaluate.py        # 评估接口
│   └── schemas.py             # Pydantic数据模型
│
├── scripts/                   # 脚本 [重构]
│   ├── run_api.py             # 启动API (支持 --mode)
│   ├── run_evaluate.py        # 运行评估
│   ├── run_ingest.py          # 运行数据摄入
│   └── run_all.sh             # 一键启动
│
├── frontend/                  # Next.js前端 [新增]
│   ├── package.json
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx           # 首页
│   │   ├── upload/            # 上传页
│   │   ├── chat/              # 问答页
│   │   └── api/               # API Routes
│   └── components/
│       ├── PDFViewer.tsx
│       ├── ChatInterface.tsx
│       └── ProgressBar.tsx
│
├── deploy/                    # 部署配置 [新增]
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   └── k8s/                   # Kubernetes配置 (可选)
│
├── data/
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   ├── embeddings/            # 向量数据库
│   ├── cache/                 # 测试模式缓存
│   └── gold_standard/         # 金标准数据
│
└── tests/                     # 集成测试 & E2E测试
    ├── integration/
    └── e2e/
```

---

## 三、配置管理设计

### 3.1 配置文件结构 (`config/base.yaml`)

```yaml
# 公共配置
app:
  name: "CUAD Assistant"
  version: "1.0.0"

models:
  llm:
    name: "Qwen/Qwen3-8B"
    device: "cuda"
    max_tokens: 4096        # 生产环境值，测试模式也保持一致
    temperature: 0.1
  
  embedding:
    name: "Qwen/Qwen3-Embedding-0.6B"
    device: "cuda"
    batch_size: 32
  
  reranker:
    name: "Qwen/Qwen3-Reranker-0.6B"
    device: "cuda"
    top_k: 20

retrieval:
  vector_db:
    type: "chroma"
    persist_directory: "${DATA_DIR}/embeddings/chroma"
    collection_name: "contracts"
  
  hybrid:
    vector_weight: 0.7
    bm25_weight: 0.3
    top_k: 50
  
  rerank:
    enabled: true
    top_k: 10

data:
  raw_path: "${DATA_DIR}/raw"
  processed_path: "${DATA_DIR}/processed"
  gold_standard_path: "${DATA_DIR}/gold_standard"
  cache_path: "${DATA_DIR}/cache"

api:
  host: "0.0.0.0"
  port: 8000

evaluation:
  ragas:
    enabled: true
    metrics: ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
  traditional:
    metrics: ["f1", "exact_match", "recall_at_k", "mrr"]
  wandb:
    enabled: true
    project: "cuad-assistant"

logging:
  level: "INFO"
  file: "./logs/app.log"
```

### 3.2 测试模式配置 (`config/test.yaml`)

```yaml
# 继承 base.yaml，覆盖测试专用配置
_inherit: base

data:
  sampling:
    enabled: true
    ratio: 0.1              # 10% 数据采样
    seed: 42                # 固定种子保证可复现
    cache_enabled: true     # 启用缓存

models:
  embedding:
    batch_size: 8           # 减小批次加速测试

retrieval:
  hybrid:
    top_k: 20               # 减少检索数量

evaluation:
  wandb:
    enabled: false          # 测试模式关闭WandB

logging:
  level: "DEBUG"
```

### 3.3 配置加载器设计要点

- 支持 `--mode test/dev/prod` 参数
- 支持环境变量覆盖 (如 `${DATA_DIR}`)
- 支持配置继承和合并 (`_inherit` 字段)
- 提供类型安全的配置访问接口

---

## 四、编写顺序（按重要性排列）

### 阶段 1：基础框架（最高优先级）

**目标**：建立可扩展的项目骨架

| 序号 | 任务 | 说明 | 依赖 |
|------|------|------|------|
| 1.1 | 配置管理系统 | `src/core/config.py` + YAML配置文件 | 无 |
| 1.2 | 项目结构重组 | 按新目录结构迁移代码 | 1.1 |
| 1.3 | 抽象接口定义 | `src/core/interfaces.py` 定义各模块接口 | 1.1 |
| 1.4 | 数据模块解耦 | 重构 `data_loader.py`，移除硬编码路径 | 1.1, 1.2 |

### 阶段 2：PDF 解析引擎（高优先级）

**目标**：替换 CSV 方案，支持真实 PDF 输入

| 序号 | 任务 | 说明 | 依赖 |
|------|------|------|------|
| 2.1 | PDF 文字提取 | Marker/Unstructured 集成 | 1.* |
| 2.2 | OCR 引擎 | PaddleOCR 集成，处理扫描件/图片 | 2.1 |
| 2.3 | 表格提取 | 保持表格结构 | 2.1 |
| 2.4 | 异常处理 | 损坏PDF/加密/OCR失败/编码问题 | 2.1-2.3 |
| 2.5 | 单元测试 | PDF模块测试覆盖 | 2.1-2.4 |

### 阶段 3：评估体系（高优先级）

**目标**：建立可量化的评估标准

| 序号 | 任务 | 说明 | 依赖 |
|------|------|------|------|
| 3.1 | 金标准格式化 | 从现有CSV生成RAGAS格式 | 1.* |
| 3.2 | RAGAS 集成 | Faithfulness, Answer Relevancy等 | 3.1 |
| 3.3 | 传统指标实现 | F1, Recall, MRR, Hit@K | 3.1 |
| 3.4 | WandB 集成 | 实验追踪和对比 | 3.2, 3.3 |
| 3.5 | 评估管道 | 自动化评估流程 | 3.1-3.4 |

### 阶段 4：API 服务重构（中优先级）

**目标**：提供稳定的后端接口

| 序号 | 任务 | 说明 | 依赖 |
|------|------|------|------|
| 4.1 | FastAPI 路由重构 | 按功能拆分路由 | 1.*, 2.* |
| 4.2 | PDF 上传接口 | 文件上传 + 异步解析 | 2.*, 4.1 |
| 4.3 | 问答接口 | 流式响应支持 | 4.1 |
| 4.4 | 评估接口 | 触发评估 + 结果查询 | 3.*, 4.1 |

### 阶段 5：前端开发（中优先级）

**目标**：提供用户交互界面

| 序号 | 任务 | 说明 | 依赖 |
|------|------|------|------|
| 5.1 | Next.js 项目初始化 | App Router + TypeScript | 无 |
| 5.2 | PDF 上传页 | 拖拽上传 + 进度显示 | 4.2, 5.1 |
| 5.3 | 问答界面 | 流式响应 + 打字机效果 | 4.3, 5.1 |
| 5.4 | PDF 预览器 | 引用跳转到页码 | 5.3 |
| 5.5 | 鉴权集成 | Next-Auth | 5.1-5.4 |

### 阶段 6：部署与运维（中优先级）

**目标**：生产环境交付

| 序号 | 任务 | 说明 | 依赖 |
|------|------|------|------|
| 6.1 | Docker 化 | API + Frontend Dockerfile | 4.*, 5.* |
| 6.2 | Docker Compose | 多服务编排 | 6.1 |
| 6.3 | 对象存储 | MinIO/S3 集成 | 6.1 |
| 6.4 | 任务队列 | BullMQ + Redis | 6.2 |
| 6.5 | 监控告警 | LangSmith/Prometheus | 6.2 |

### 阶段 7：测试完善（持续进行）

**目标**：保证代码质量

| 序号 | 任务 | 说明 | 依赖 |
|------|------|------|------|
| 7.1 | 单元测试 | 每个模块 | 各阶段并行 |
| 7.2 | 集成测试 | 模块间交互 | 阶段 1-4 完成后 |
| 7.3 | E2E 测试 | 完整流程 | 阶段 5 完成后 |

---

## 五、执行时间线（建议）

```
阶段1 ████████░░░░░░░░░░░░░░░░░░░░░░  基础框架
阶段2     ████████████░░░░░░░░░░░░░░  PDF解析
阶段3         ████████████░░░░░░░░░░  评估体系
阶段4             ████████░░░░░░░░░░  API重构
阶段5                 ████████████░░  前端开发
阶段6                     ████████░░  部署运维
阶段7 ══════════════════════════════  测试(持续)
      ─────────────────────────────────────────►
```

---

## 六、快速启动命令设计

```bash
# 开发模式
python scripts/run_api.py --mode dev

# 测试模式 (10%数据 + 缓存)
python scripts/run_api.py --mode test

# 生产模式
python scripts/run_api.py --mode prod

# 运行评估
python scripts/run_evaluate.py --mode test --output ./results

# Docker一键启动
docker-compose -f deploy/docker-compose.dev.yml up
```

---

## 七、RAGAS 评估集成

### 7.1 数据格式映射

| 现有字段 (gold_answers.csv) | RAGAS 字段 | 说明 |
|----------------------------|------------|------|
| `query` | `question` | 直接可用 |
| `gold_answer_text` | `ground_truth` | 直接可用 |
| `gold_chunk_ids` | 用于 Retrieval 指标 | 检索金标准 |
| - | `answer` | 运行时由 LLM 生成 |
| - | `contexts` | 运行时由检索器返回 |

### 7.2 评估指标

**RAGAS 指标**:
- Faithfulness (忠实度)
- Answer Relevancy (答案相关性)
- Context Precision (上下文精确度)
- Context Recall (上下文召回率)

**传统指标**:
- F1 Score
- Exact Match
- Recall@K
- MRR (Mean Reciprocal Rank)
- Hit@K

### 7.3 评估流程

```
1. 加载 gold_answers.csv (已有)
          ↓
2. 对每个 query 运行 RAG 管道
   - 检索 → contexts
   - 生成 → answer
          ↓
3. 组装 RAGAS Dataset
          ↓
4. 计算指标
   - RAGAS 指标
   - 传统指标
          ↓
5. 上报 WandB + 生成报告
```

---

## 八、PDF 解析异常处理

### 8.1 需要处理的异常类型

| 异常类型 | 说明 | 处理策略 |
|----------|------|----------|
| 损坏的 PDF | 文件头损坏、页面缺失 | 记录错误，跳过或降级处理 |
| 加密 PDF | 密码保护文档 | 提示用户提供密码或跳过 |
| OCR 失败 | 图片质量过低无法识别 | 降级返回原始图片描述 |
| 编码问题 | 乱码、特殊字符 | 字符集自动检测和转换 |
| 超大文件 | 内存溢出风险 | 分页处理、流式解析 |
| 多语言混合 | 中英文混排 | 多语言OCR模型支持 |

### 8.2 异常类定义

```python
class PDFParseError(Exception):
    """PDF解析基础异常"""
    pass

class PDFCorruptedError(PDFParseError):
    """PDF文件损坏"""
    pass

class PDFEncryptedError(PDFParseError):
    """PDF加密未解锁"""
    pass

class OCRFailedError(PDFParseError):
    """OCR识别失败"""
    pass

class PDFTooLargeError(PDFParseError):
    """PDF文件过大"""
    pass
```

---

## 九、技术栈汇总

| 层级 | 技术选型 |
|------|----------|
| 前端 | Next.js 14+, TypeScript, Tailwind CSS |
| 鉴权 | Next-Auth.js / Clerk |
| 任务队列 | BullMQ (Redis) |
| 元数据库 | PostgreSQL + Prisma ORM |
| PDF解析 | Marker, Unstructured.io, PaddleOCR |
| 分块策略 | LangChain RecursiveCharacterTextSplitter |
| LLM推理 | vLLM (Qwen3-8B) |
| Embedding | Qwen3-Embedding-0.6B |
| Reranker | Qwen3-Reranker-0.6B |
| 向量库 | ChromaDB (可升级 Qdrant) |
| API服务 | FastAPI |
| 容器化 | Docker + Docker Compose |
| 对象存储 | MinIO / AWS S3 |
| 链路追踪 | LangSmith / Phoenix |
| 评估框架 | RAGAS + WandB |

---

## 十、注意事项

1. **配置管理优先**：所有路径和参数通过配置文件管理，避免硬编码
2. **测试模式一致性**：`max_tokens` 等关键参数在测试模式下保持与生产一致
3. **渐进式开发**：先用 CSV 跑通流程，再逐步替换为 PDF 解析
4. **金标准复用**：`master_clauses.csv` 可直接用于生成 RAGAS 评估数据
5. **模块化设计**：每个模块独立目录 + 单元测试，便于维护和扩展

---

## 十一、PDF 坐标溯源与高亮 (The "Foxit" Feature)

> **核心价值**：这是文档类 RAG 的核心体验，也是 Foxit 最看重的技术点。仅返回文本引用是不够的，需要传递 Bounding Box (bbox) 数据实现精确定位。

### 11.1 数据流增强

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐    ┌─────────────────┐
│  PDF Parser │───▶│ Extract Text +  │───▶│ ChromaDB Store  │───▶│ LLM + Cite  │───▶│ Frontend PDF    │
│  (Marker)   │    │ BBox Coords     │    │ with Metadata   │    │ Generation  │    │ Highlight Layer │
└─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘    └─────────────────┘
```

### 11.2 坐标数据结构

```python
# 每个 Token/句子的坐标信息
@dataclass
class BoundingBox:
    page_num: int      # 页码 (从0开始)
    x0: float          # 左上角 x
    y0: float          # 左上角 y
    x1: float          # 右下角 x
    y1: float          # 右下角 y

@dataclass  
class ChunkWithBBox:
    chunk_id: str
    text: str
    bbox_list: List[BoundingBox]  # 一个chunk可能跨多个bbox
    file_name: str
    page_range: Tuple[int, int]   # 起止页码
```

### 11.3 各层实现要点

| 层级 | 实现要点 |
|------|----------|
| **解析层 (Parser)** | 使用 `Marker` 或 `PyMuPDF` 时，必须提取每个 Token 或句子的坐标 `[page_num, x0, y0, x1, y1]` |
| **存储层 (ChromaDB)** | 在向量数据库的 `metadata` 中，必须包含该切片对应的 `bbox_list` (JSON序列化存储) |
| **推理层 (LLM)** | 返回答案时，附带引用标记 `[1]`, `[2]` 等，每个引用对应一个 `chunk_id` |
| **交互层 (Frontend)** | PDF Viewer 接收到引用时，解析对应的 bbox 坐标，在 PDF Canvas 上绘制半透明黄色矩形 |

### 11.4 前端高亮实现

```typescript
// 前端 PDF 高亮组件示例
interface Citation {
  ref_id: string;        // "[1]"
  chunk_id: string;
  bbox_list: BoundingBox[];
}

// 使用 react-pdf 或 pdf.js 绘制高亮
const renderHighlight = (citation: Citation, scale: number) => {
  return citation.bbox_list.map((bbox, idx) => (
    <div
      key={idx}
      style={{
        position: 'absolute',
        left: bbox.x0 * scale,
        top: bbox.y0 * scale,
        width: (bbox.x1 - bbox.x0) * scale,
        height: (bbox.y1 - bbox.y0) * scale,
        backgroundColor: 'rgba(255, 255, 0, 0.3)',
        pointerEvents: 'none',
      }}
    />
  ));
};
```

### 11.5 ChromaDB Metadata 存储格式

```python
# 存入向量库时的 metadata 结构
metadata = {
    "chunk_id": "doc1::clause1::0::0::100:500",
    "file_name": "contract_2024.pdf",
    "clause_type": "Termination",
    "page_start": 3,
    "page_end": 4,
    "bbox_json": json.dumps([
        {"page": 3, "x0": 50.0, "y0": 100.0, "x1": 550.0, "y1": 150.0},
        {"page": 3, "x0": 50.0, "y0": 150.0, "x1": 550.0, "y1": 200.0},
    ])
}
```

---

## 十二、表格处理专项策略 (Table Handling)

> **核心问题**：合同中的表格（如财务数据、交付里程碑）往往非常复杂，简单的 Markdown 转换容易丢失行列对齐信息，且检索时表格语义容易被打散。

### 12.1 表格增强策略流程

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ PDF Parser  │───▶│ 检测到表格      │───▶│ LLM 生成摘要    │───▶│ 摘要 Embedding  │
│ 版面分析    │    │ <table>...</>   │    │ (自然语言描述)  │    │ 索引存储        │
└─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                          │
                                                                          ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ LLM 阅读    │◀───│ 返回完整表格    │◀───│ 检索命中摘要    │◀───│ 用户查询        │
│ 完整表格    │    │ Markdown/HTML   │    │ chunk           │    │                 │
└─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 12.2 表格数据结构

```python
@dataclass
class TableChunk:
    table_id: str
    summary: str                    # LLM生成的摘要 (用于Embedding)
    raw_markdown: str               # 原始Markdown表格
    raw_html: str                   # 原始HTML表格 (保留格式)
    headers: List[str]              # 表头列名
    row_count: int                  # 行数
    col_count: int                  # 列数
    bbox: BoundingBox               # 表格位置
    context_before: str             # 表格前的上下文 (如标题)
    context_after: str              # 表格后的上下文
```

### 12.3 表格摘要生成 Prompt

```python
TABLE_SUMMARY_PROMPT = """
请为以下表格生成一段简洁的自然语言摘要（50-100字），描述：
1. 这是什么类型的表格（付款计划、里程碑、费用明细等）
2. 包含哪些关键信息（时间范围、金额范围、责任方等）
3. 有多少行/列数据

表格内容：
{table_markdown}

表格前文：
{context_before}

请直接输出摘要，不要包含其他内容。
"""
```

### 12.4 分块策略规则

| 规则 | 说明 |
|------|------|
| **不切分原则** | `<table>...</table>` 强制作为一个整体 chunk，禁止中间切断 |
| **最大限制** | 单表格超过 `max_table_tokens` (默认2000) 时，按行分组切分 |
| **上下文保留** | 表格 chunk 自动包含前后各 100 字符的上下文 |
| **双索引** | 同时存储摘要向量 (用于检索) 和原始表格 (用于生成) |

### 12.5 ChromaDB 表格存储

```python
# 表格存储时的特殊处理
table_metadata = {
    "chunk_id": "doc1::table::page5::0",
    "chunk_type": "table",           # 标记为表格类型
    "summary": "2024年Q1-Q4付款节点表，共4个里程碑，总金额$500,000",
    "raw_content": "| 里程碑 | 日期 | 金额 |\n|---|---|---|\n| M1 | 2024-03 | $100,000 |...",
    "row_count": 5,
    "col_count": 3,
    "bbox_json": "...",
}

# 检索时：用 summary 的 embedding 进行相似度计算
# 生成时：将 raw_content 完整传给 LLM
```

---

## 十三、语义缓存 (Semantic Caching)

> **核心价值**：合同审查场景中，用户经常问重复问题（如"合同截止日期是哪天？"）。语义缓存可以显著降低延迟和成本。

### 13.1 查询优化流程

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ User Query  │───▶│ Query Embedding │───▶│ Redis Semantic  │
│             │    │                 │    │ Cache Lookup    │
└─────────────┘    └─────────────────┘    └─────────────────┘
                                                   │
                                    ┌──────────────┴──────────────┐
                                    │                             │
                              [Cache Hit]                   [Cache Miss]
                              similarity > 0.95             similarity < 0.95
                                    │                             │
                                    ▼                             ▼
                          ┌─────────────────┐           ┌─────────────────┐
                          │ Return Cached   │           │ Full RAG        │
                          │ Answer Directly │           │ Pipeline        │
                          └─────────────────┘           └─────────────────┘
                                                                  │
                                                                  ▼
                                                        ┌─────────────────┐
                                                        │ Cache New       │
                                                        │ Query-Answer    │
                                                        └─────────────────┘
```

### 13.2 缓存数据结构

```python
@dataclass
class CacheEntry:
    query_text: str
    query_embedding: List[float]    # 用于相似度计算
    answer_text: str
    contexts: List[str]             # 检索到的上下文
    citations: List[Citation]       # 引用信息
    document_id: str                # 关联的文档ID
    created_at: datetime
    hit_count: int                  # 命中次数统计
    ttl: int                        # 过期时间(秒)
```

### 13.3 配置参数

```yaml
# config/base.yaml 增加缓存配置
cache:
  semantic:
    enabled: true
    backend: "redis"                # redis / memory
    similarity_threshold: 0.95      # 相似度阈值
    ttl_seconds: 86400              # 24小时过期
    max_entries: 10000              # 最大缓存条目
    embedding_dim: 1024             # Embedding维度
  
  # Redis 连接配置
  redis:
    host: "localhost"
    port: 6379
    db: 0
    password: "${REDIS_PASSWORD}"
```

### 13.4 缓存实现示例

```python
class SemanticCache:
    def __init__(self, redis_client, embedding_model, threshold=0.95):
        self.redis = redis_client
        self.embedder = embedding_model
        self.threshold = threshold
    
    def get(self, query: str, document_id: str) -> Optional[CacheEntry]:
        """查找语义相似的缓存"""
        query_emb = self.embedder.encode(query)
        
        # 获取该文档的所有缓存条目
        cache_keys = self.redis.keys(f"cache:{document_id}:*")
        
        for key in cache_keys:
            entry = self._deserialize(self.redis.get(key))
            similarity = cosine_similarity(query_emb, entry.query_embedding)
            
            if similarity >= self.threshold:
                entry.hit_count += 1
                self.redis.set(key, self._serialize(entry))
                return entry
        
        return None
    
    def set(self, query: str, answer: str, contexts: List[str], 
            citations: List[Citation], document_id: str):
        """存入新缓存"""
        entry = CacheEntry(
            query_text=query,
            query_embedding=self.embedder.encode(query),
            answer_text=answer,
            contexts=contexts,
            citations=citations,
            document_id=document_id,
            created_at=datetime.now(),
            hit_count=0,
            ttl=self.config.ttl_seconds,
        )
        key = f"cache:{document_id}:{hash(query)}"
        self.redis.setex(key, entry.ttl, self._serialize(entry))
```

### 13.5 缓存失效策略

| 场景 | 失效策略 |
|------|----------|
| 文档更新 | 清除该 `document_id` 下所有缓存 |
| TTL 过期 | Redis 自动过期删除 |
| 用户点踩 | 立即删除该缓存条目 |
| 容量超限 | LRU 策略淘汰最少命中的条目 |

---

## 十四、用户反馈闭环 (Feedback Loop)

> **核心价值**：RAGAS 提供离线评估，但系统上线后需要利用用户行为来持续优化模型。这是从"Demo"到"生产系统"的关键闭环。

### 14.1 数据飞轮流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户交互层                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │   👍 点赞   │    │   👎 点踩   │    │  ✏️ 修正    │                      │
│  └─────────────┘    └─────────────┘    └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              反馈存储层                                      │
│  PostgreSQL: feedback_logs 表                                               │
│  - query, answer, user_rating, corrected_answer, feedback_type, timestamp   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据分析层                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ Bad Case 聚类   │    │ 错误类型统计    │    │ 问题分布分析    │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              模型优化层                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ Golden Dataset  │    │ RAGAS 回归测试  │    │ Reranker 微调   │          │
│  │ 更新            │    │ 定期触发        │    │ Prompt 优化     │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 14.2 数据库 Schema

```sql
-- PostgreSQL 反馈日志表
CREATE TABLE feedback_logs (
    id SERIAL PRIMARY KEY,
    
    -- 查询信息
    query_text TEXT NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    
    -- 系统响应
    answer_text TEXT NOT NULL,
    contexts JSONB,                    -- 检索到的上下文
    citations JSONB,                   -- 引用信息
    
    -- 用户反馈
    rating VARCHAR(20) NOT NULL,       -- 'positive' / 'negative'
    feedback_type VARCHAR(50),         -- 'irrelevant' / 'hallucination' / 'incomplete' / 'wrong_citation'
    corrected_answer TEXT,             -- 用户修正的答案
    user_comment TEXT,                 -- 用户备注
    
    -- 元数据
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    response_time_ms INT,
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- 索引
    INDEX idx_document_id (document_id),
    INDEX idx_rating (rating),
    INDEX idx_created_at (created_at)
);

-- 金标准数据集表 (从反馈中提取)
CREATE TABLE golden_dataset (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    gold_answer TEXT NOT NULL,
    gold_contexts JSONB,
    source VARCHAR(50),                -- 'manual' / 'feedback' / 'cuad'
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 14.3 反馈类型定义

```python
class FeedbackType(Enum):
    IRRELEVANT = "irrelevant"           # 答案与问题不相关
    HALLUCINATION = "hallucination"     # 答案包含幻觉/捏造内容
    INCOMPLETE = "incomplete"           # 答案不完整
    WRONG_CITATION = "wrong_citation"   # 引用位置错误
    OUTDATED = "outdated"               # 答案过时
    FORMAT_ERROR = "format_error"       # 格式/表达问题
    OTHER = "other"                     # 其他问题
```

### 14.4 前端反馈组件

```typescript
interface FeedbackProps {
  queryId: string;
  answer: string;
  onFeedback: (feedback: Feedback) => void;
}

const FeedbackButtons: React.FC<FeedbackProps> = ({ queryId, answer, onFeedback }) => {
  const [showModal, setShowModal] = useState(false);
  
  return (
    <div className="flex gap-2 mt-2">
      <button 
        onClick={() => onFeedback({ rating: 'positive' })}
        className="p-2 hover:bg-green-100 rounded"
      >
        👍
      </button>
      
      <button 
        onClick={() => setShowModal(true)}
        className="p-2 hover:bg-red-100 rounded"
      >
        👎
      </button>
      
      {showModal && (
        <FeedbackModal
          onSubmit={(data) => {
            onFeedback({
              rating: 'negative',
              feedback_type: data.type,
              corrected_answer: data.correction,
              comment: data.comment,
            });
            setShowModal(false);
          }}
          onClose={() => setShowModal(false)}
        />
      )}
    </div>
  );
};
```

### 14.5 自动化回归测试

```yaml
# 定时任务配置
scheduled_tasks:
  regression_test:
    cron: "0 2 * * 0"              # 每周日凌晨2点
    description: "基于Golden Dataset运行RAGAS回归测试"
    steps:
      - load_golden_dataset
      - run_rag_pipeline
      - calculate_ragas_metrics
      - compare_with_baseline
      - alert_if_degraded
      - upload_to_wandb

  bad_case_analysis:
    cron: "0 3 * * *"              # 每天凌晨3点
    description: "分析前一天的负反馈"
    steps:
      - aggregate_negative_feedback
      - cluster_by_error_type
      - generate_report
      - notify_team
```

### 14.6 优化闭环触发条件

| 触发条件 | 执行动作 |
|----------|----------|
| 负反馈率 > 10% | 发送告警，人工复核 |
| 累计 50 条同类型错误 | 自动添加到 Golden Dataset，触发回归测试 |
| RAGAS 指标下降 > 5% | 阻止新版本部署，回滚到上一版本 |
| 每周固定时间 | 批量分析负反馈，优化 Prompt 或 Reranker |

---

## 十五、更新后的目录结构

```
cuad-assistant/
├── config/
│   ├── base.yaml              # 公共配置 (含缓存配置)
│   ├── dev.yaml
│   ├── test.yaml
│   └── prod.yaml
│
├── src/
│   ├── core/
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── interfaces.py
│   │
│   ├── pdf/
│   │   ├── parser.py
│   │   ├── ocr_engine.py
│   │   ├── table_extractor.py
│   │   ├── layout_analyzer.py
│   │   ├── bbox_extractor.py      # [新增] BBox坐标提取
│   │   └── exceptions.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── table_chunker.py       # [新增] 表格专用分块
│   │   └── gold_standard.py
│   │
│   ├── rag/
│   │   ├── embedding.py
│   │   ├── retrieval.py
│   │   ├── reranker.py
│   │   └── semantic_cache.py      # [新增] 语义缓存
│   │
│   ├── inference/
│   │   ├── llm_inference.py
│   │   └── citation_parser.py     # [新增] 引用解析
│   │
│   ├── evaluation/
│   │   ├── ragas_adapter.py
│   │   ├── metrics.py
│   │   ├── wandb_reporter.py
│   │   └── pipeline.py
│   │
│   ├── feedback/                  # [新增] 反馈模块
│   │   ├── __init__.py
│   │   ├── collector.py           # 反馈收集
│   │   ├── analyzer.py            # 反馈分析
│   │   ├── golden_updater.py      # 金标准更新
│   │   └── regression_runner.py   # 回归测试
│   │
│   └── utils/
│       ├── prompting.py
│       ├── plot.py
│       └── model_loading.py
│
├── api/
│   ├── main.py
│   ├── routes/
│   │   ├── pdf.py
│   │   ├── query.py
│   │   ├── evaluate.py
│   │   └── feedback.py            # [新增] 反馈接口
│   └── schemas.py
│
├── frontend/
│   ├── app/
│   │   ├── chat/
│   │   └── upload/
│   └── components/
│       ├── PDFViewer.tsx
│       ├── HighlightLayer.tsx     # [新增] 高亮覆盖层
│       ├── ChatInterface.tsx
│       ├── FeedbackButtons.tsx    # [新增] 反馈按钮
│       └── FeedbackModal.tsx      # [新增] 反馈弹窗
│
├── deploy/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
└── data/
    ├── raw/
    ├── processed/
    ├── embeddings/
    ├── cache/
    ├── gold_standard/
    └── feedback/                  # [新增] 反馈数据
```

---

## 十六、更新后的技术栈汇总

| 层级 | 技术选型 |
|------|----------|
| 前端 | Next.js 14+, TypeScript, Tailwind CSS |
| PDF高亮 | react-pdf / pdf.js + Canvas Overlay |
| 鉴权 | Next-Auth.js / Clerk |
| 任务队列 | BullMQ (Redis) |
| 语义缓存 | Redis + 自定义相似度检索 / GPTCache |
| 元数据库 | PostgreSQL + Prisma ORM |
| PDF解析 | Marker (含BBox), PyMuPDF, PaddleOCR |
| 表格处理 | Marker Table Detection + LLM Summary |
| 分块策略 | LangChain RecursiveCharacterTextSplitter |
| LLM推理 | vLLM (Qwen3-8B) |
| Embedding | Qwen3-Embedding-0.6B |
| Reranker | Qwen3-Reranker-0.6B |
| 向量库 | ChromaDB (含BBox Metadata) |
| API服务 | FastAPI |
| 容器化 | Docker + Docker Compose |
| 对象存储 | MinIO / AWS S3 |
| 链路追踪 | LangSmith / Phoenix |
| 评估框架 | RAGAS + WandB |
| 反馈系统 | PostgreSQL + 定时任务 |

---

## 十七、更新后的编写顺序

> 将新增的4个关键维度整合到原有阶段中

### 阶段 1：基础框架（最高优先级）
- 1.1 配置管理系统 (含缓存配置)
- 1.2 项目结构重组
- 1.3 抽象接口定义
- 1.4 数据模块解耦

### 阶段 2：PDF 解析引擎（高优先级）
- 2.1 PDF 文字提取
- 2.2 **BBox 坐标提取** ⭐ 新增
- 2.3 OCR 引擎
- 2.4 **表格检测与摘要生成** ⭐ 新增
- 2.5 异常处理
- 2.6 单元测试

### 阶段 3：评估体系（高优先级）
- 3.1 金标准格式化
- 3.2 RAGAS 集成
- 3.3 传统指标实现
- 3.4 WandB 集成
- 3.5 评估管道

### 阶段 4：API 服务重构（中优先级）
- 4.1 FastAPI 路由重构
- 4.2 PDF 上传接口 (含BBox元数据)
- 4.3 问答接口 (含引用坐标返回)
- 4.4 **语义缓存层** ⭐ 新增
- 4.5 评估接口

### 阶段 5：前端开发（中优先级）
- 5.1 Next.js 项目初始化
- 5.2 PDF 上传页
- 5.3 问答界面
- 5.4 **PDF 高亮预览器** ⭐ 新增
- 5.5 鉴权集成

### 阶段 6：部署与运维（中优先级）
- 6.1 Docker 化
- 6.2 Docker Compose (含Redis)
- 6.3 对象存储
- 6.4 任务队列
- 6.5 监控告警

### 阶段 7：用户反馈闭环（后期优先级）⭐ 新增阶段
- 7.1 反馈收集接口
- 7.2 反馈存储 (PostgreSQL)
- 7.3 前端反馈组件
- 7.4 Bad Case 分析管道
- 7.5 Golden Dataset 自动更新
- 7.6 回归测试定时任务

### 阶段 8：测试完善（持续进行）
- 8.1 单元测试
- 8.2 集成测试
- 8.3 E2E 测试
