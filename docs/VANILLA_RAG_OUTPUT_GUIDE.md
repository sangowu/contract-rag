# Vanilla RAG 测试输出指南

**测试脚本**: `scripts/run_vanilla_rag.py`

---

## 📁 输出位置

### 默认输出目录
```
results/vanilla_rag/
├── vanilla_rag_results.csv        # 详细结果（每个样本）
└── vanilla_rag_results_summary.json   # 汇总统计
```

### 自定义输出目录
```bash
# 指定输出位置
python scripts/run_vanilla_rag.py --output-dir results/my_test
```

---

## 📊 生成的文件

### 1. `vanilla_rag_results.csv` - 详细结果

**每一行代表一个测试样本**，包含以下列：

#### 基础信息列（8列）
| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `query` | string | 用户查询 | "What is the termination clause?" |
| `file_name` | string | 合同文件名 | "AEMETIS INC_04_09_2010.pdf" |
| `gold_answer` | string | 金标准答案 | "The agreement may be terminated..." |
| `pred_answer` | string | 模型生成答案 | "The contract states that..." |
| `answer_type` | string | 答案类型 | "text" / "boolean" / "none" |
| `clause_type` | string | 条款类型 | "Termination" / "Payment" |
| `retrieved_ids` | list | 检索到的 chunk IDs | "[123, 456, 789]" |
| `gold_ids` | list | 金标准 chunk IDs | "[123, 789]" |

#### 检索指标列（16列）
| 列名 | 类型 | 说明 | 范围 |
|------|------|------|------|
| `hit@1` | float | Top-1 命中率 | 0.0-1.0 |
| `hit@3` | float | Top-3 命中率 | 0.0-1.0 |
| `hit@5` | float | Top-5 命中率 | 0.0-1.0 |
| `hit@10` | float | Top-10 命中率 | 0.0-1.0 |
| `mrr@1` | float | Top-1 平均倒数排名 | 0.0-1.0 |
| `mrr@3` | float | Top-3 平均倒数排名 | 0.0-1.0 |
| `mrr@5` | float | Top-5 平均倒数排名 | 0.0-1.0 |
| `mrr@10` | float | Top-10 平均倒数排名 | 0.0-1.0 |
| `recall@1` | float | Top-1 召回率 | 0.0-1.0 |
| `recall@3` | float | Top-3 召回率 | 0.0-1.0 |
| `recall@5` | float | Top-5 召回率 | 0.0-1.0 |
| `recall@10` | float | Top-10 召回率 | 0.0-1.0 |
| `precision@1` | float | Top-1 精确率 | 0.0-1.0 |
| `precision@3` | float | Top-3 精确率 | 0.0-1.0 |
| `precision@5` | float | Top-5 精确率 | 0.0-1.0 |
| `precision@10` | float | Top-10 精确率 | 0.0-1.0 |

#### 答案质量指标列（4列）
| 列名 | 类型 | 说明 | 范围 |
|------|------|------|------|
| `f1_score` | float | F1 分数 | 0.0-1.0 |
| `exact_match` | float | 完全匹配 | 0.0 或 1.0 |
| `precision` | float | 答案精确率 | 0.0-1.0 |
| `recall` | float | 答案召回率 | 0.0-1.0 |

#### 简化 RAGAS 指标列（3列）
| 列名 | 类型 | 说明 | 范围 |
|------|------|------|------|
| `simple_context_coverage` | float | 上下文覆盖率 | 0.0-1.0 |
| `simple_answer_context_overlap` | float | 答案-上下文重叠度 | 0.0-1.0 |
| `simple_ground_truth_coverage` | float | 金标准覆盖率 | 0.0-1.0 |

#### 性能指标列（3列）
| 列名 | 类型 | 说明 | 单位 |
|------|------|------|------|
| `retrieval_time_ms` | float | 检索耗时 | 毫秒 |
| `generation_time_ms` | float | 生成耗时 | 毫秒 |
| `total_time_ms` | float | 端到端耗时 | 毫秒 |

**总计**: 约 **37 列**

---

### 2. `vanilla_rag_results_summary.json` - 汇总统计

**JSON 格式的整体统计信息**：

```json
{
  "total_samples": 510,
  "timestamp": "2024-01-15T10:30:45",
  
  "retrieval_metrics": {
    "hit@10_mean": 0.452,
    "hit@10_std": 0.498,
    "mrr_mean": 0.380,
    "mrr_std": 0.402,
    "recall@10_mean": 0.521,
    "recall@10_std": 0.445
  },
  
  "answer_metrics": {
    "f1_score_mean": 0.312,
    "f1_score_std": 0.298,
    "exact_match": 0.085,
    "precision_mean": 0.345,
    "recall_mean": 0.298
  },
  
  "performance_metrics": {
    "avg_retrieval_time_ms": 125.3,
    "avg_generation_time_ms": 1850.2,
    "avg_total_time_ms": 2300.5,
    "throughput_samples_per_min": 12.5,
    "p50_total_time_ms": 2100,
    "p95_total_time_ms": 3500,
    "p99_total_time_ms": 4200
  },
  
  "by_answer_type": {
    "text": {
      "count": 380,
      "f1_score_mean": 0.345,
      "hit@10_mean": 0.468
    },
    "boolean": {
      "count": 100,
      "f1_score_mean": 0.220,
      "hit@10_mean": 0.410
    },
    "none": {
      "count": 30,
      "f1_score_mean": 0.0,
      "hit@10_mean": 0.367
    }
  },
  
  "by_clause_type": {
    "Termination": {
      "count": 45,
      "f1_score_mean": 0.398,
      "hit@10_mean": 0.533
    },
    "Payment": {
      "count": 38,
      "f1_score_mean": 0.312,
      "hit@10_mean": 0.447
    }
    // ... 其他条款类型
  }
}
```

---

## 🖥️ 控制台输出

### 运行时输出

```
==================================================
Starting Vanilla RAG Evaluation
==================================================
  Total samples: 510
  Top K: 10
  Output: results/vanilla_rag
==================================================

Loading embedding model...
✅ Embedding model loaded

Evaluating: 100%|████████████| 510/510 [25:30<00:00, 3.0s/it]

Calculating metrics...

==================================================
Evaluation Summary
==================================================

📊 Retrieval Metrics (Mean ± Std):
------------------------------------
  Hit@1:        0.123 ± 0.329
  Hit@3:        0.287 ± 0.453
  Hit@5:        0.365 ± 0.482
  Hit@10:       0.452 ± 0.498
  
  MRR@10:       0.380 ± 0.402
  
  Recall@1:     0.145 ± 0.352
  Recall@3:     0.334 ± 0.472
  Recall@5:     0.421 ± 0.494
  Recall@10:    0.521 ± 0.445
  
  Precision@10: 0.089 ± 0.125

📝 Answer Quality Metrics:
------------------------------------
  F1 Score:     0.312 ± 0.298
  Exact Match:  8.5%
  Precision:    0.345 ± 0.315
  Recall:       0.298 ± 0.289

⚡ Performance Metrics:
------------------------------------
  Avg Retrieval Time:   125.3 ms
  Avg Generation Time:  1850.2 ms
  Avg Total Time:       2300.5 ms
  
  Throughput:           12.5 samples/min
  
  P50 Latency:          2100 ms
  P95 Latency:          3500 ms
  P99 Latency:          4200 ms

==================================================

✅ Results saved to: results/vanilla_rag/vanilla_rag_results.csv
✅ Summary saved to: results/vanilla_rag/vanilla_rag_results_summary.json

==================================================
Vanilla RAG Evaluation Complete!
==================================================
```

---

## ❓ 是否使用前端？

### ⭕ **不使用前端**

`run_vanilla_rag.py` 是一个**纯后端测试脚本**，特点：

| 特性 | 说明 |
|------|------|
| **运行方式** | 命令行脚本 |
| **数据来源** | 直接从 CSV 文件加载 |
| **评估方式** | 自动批量评估 |
| **结果输出** | CSV + JSON 文件 |
| **无需前端** | ✅ 完全独立运行 |
| **无需 API** | ✅ 直接调用后端函数 |

### 📊 测试流程

```
CSV 数据 (gold_answers.csv)
    ↓
加载到内存
    ↓
逐个样本测试
    ├─→ vanilla_retrieve()    ← 直接调用
    └─→ vanilla_generate()    ← 直接调用
    ↓
计算评估指标
    ↓
保存 CSV + JSON
```

### 🔄 与前端测试的区别

| 方面 | Vanilla RAG 脚本 | 前端测试 |
|------|----------------|---------|
| **运行环境** | Python 命令行 | Web 浏览器 |
| **数据输入** | CSV 文件 | 用户手动输入 |
| **测试规模** | 批量（500+ 样本） | 单个查询 |
| **评估方式** | 自动化指标 | 人工检查 |
| **输出格式** | CSV + JSON | 网页显示 |
| **用途** | Benchmark 测试 | 用户体验测试 |

---

## 📋 使用示例

### 示例 1: 快速测试（10 样本）

```bash
python scripts/run_vanilla_rag.py --mode test --max-samples 10
```

**输出**:
```
results/vanilla_rag/
├── vanilla_rag_results.csv         (10 行数据)
└── vanilla_rag_results_summary.json (统计信息)
```

### 示例 2: 完整测试

```bash
python scripts/run_vanilla_rag.py --mode test
```

**输出**:
```
results/vanilla_rag/
├── vanilla_rag_results.csv         (510 行数据)
└── vanilla_rag_results_summary.json
```

### 示例 3: 自定义输出

```bash
python scripts/run_vanilla_rag.py \
  --mode test \
  --max-samples 100 \
  --top-k 20 \
  --output-dir results/test_20240115
```

**输出**:
```
results/test_20240115/
├── vanilla_rag_results.csv
└── vanilla_rag_results_summary.json
```

---

## 📈 结果分析

### 查看 CSV 结果

```bash
# 查看前 10 行
head -10 results/vanilla_rag/vanilla_rag_results.csv

# 使用 pandas 分析
python -c "
import pandas as pd
df = pd.read_csv('results/vanilla_rag/vanilla_rag_results.csv')
print(df.describe())
print('\nHit@10 by clause type:')
print(df.groupby('clause_type')['hit@10'].mean())
"
```

### 查看 JSON 摘要

```bash
# 格式化查看
cat results/vanilla_rag/vanilla_rag_results_summary.json | python -m json.tool

# 提取关键指标
python -c "
import json
with open('results/vanilla_rag/vanilla_rag_results_summary.json') as f:
    data = json.load(f)
print(f'Hit@10: {data[\"retrieval_metrics\"][\"hit@10_mean\"]:.3f}')
print(f'F1 Score: {data[\"answer_metrics\"][\"f1_score_mean\"]:.3f}')
print(f'Throughput: {data[\"performance_metrics\"][\"throughput_samples_per_min\"]:.1f} samples/min')
"
```

---

## 🔍 与其他测试方式对比

### 1. Vanilla RAG 脚本（本脚本）
```bash
python scripts/run_vanilla_rag.py --mode test
```
- ✅ 自动化批量测试
- ✅ 完整评估指标
- ✅ 适合 Benchmark
- ❌ 无可视化界面

### 2. API + 评估脚本
```bash
# 先启动 API
python scripts/run_api.py --mode dev

# 再运行评估
python scripts/run_evaluation.py --mode test --api-url http://localhost:8000
```
- ✅ 测试 API 接口
- ✅ 更接近生产环境
- ⚠️ 需要两步操作

### 3. 前端测试
```bash
# 启动后端
python scripts/run_api.py --mode dev

# 启动前端
cd frontend && npm run dev
```
- ✅ 用户体验测试
- ✅ 可视化结果
- ❌ 无法批量测试
- ❌ 无自动评估指标

