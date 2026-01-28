# EvaluationPipeline 使用指南

**新功能**: 自动绘图 + 版本管理已集成到评估管道

---

## 🎯 核心改动

| 功能 | 说明 |
|------|------|
| **自动绘图** | `save_results()` 时自动生成图表 |
| **版本管理** | 通过 `experiment_name` + `experiment_version` 管理 |
| **统一输出** | CSV + JSON + 图表一键生成 |

---

## 📊 使用方式

### 方式 1: 使用 EvaluationPipeline 类

```python
from src.evaluation import EvaluationPipeline, EvaluationConfig

# 配置
config = EvaluationConfig(
    k_values=[1, 3, 5, 10],
    plot_enabled=True,           # 启用自动绘图
    plot_include_performance=True,
)

# 创建管道（带版本管理）
pipeline = EvaluationPipeline(
    config=config,
    experiment_name="baseline",   # 实验名称
    experiment_version="v1",      # 版本号
)

# 评估
pipeline.evaluate_batch(eval_data)

# 保存（自动生成 CSV + JSON + 图表）
output_files = pipeline.save_results()
# 返回: {"csv": "...", "summary": "...", "plots_dir": "..."}

# 打印摘要
pipeline.print_summary()
```

---

### 方式 2: 使用便捷函数

```python
from src.evaluation import evaluate_rag_results, quick_evaluate

# 完整评估
summary = evaluate_rag_results(
    results=eval_data,
    experiment_name="my_test",
    experiment_version="v2",
    plot_enabled=True,
)

# 快速评估
pipeline = quick_evaluate(
    results=eval_data,
    name="quick_test",
    version="v1",
)
```

---

### 方式 3: 使用脚本

```bash
# Vanilla RAG 测试（自动版本管理和绘图）
python scripts/run_vanilla_rag.py \
  --mode test \
  --experiment-name "baseline"

# API 评估脚本
python scripts/run_evaluation.py \
  --mode test \
  --experiment-name "api_test" \
  --experiment-version "v1"

# 禁用绘图
python scripts/run_evaluation.py \
  --mode test \
  --no-plots
```

---

## 📁 输出文件结构

```
results/
├── evaluation/                         # CSV 数据目录
│   ├── v1_baseline.csv                # 详细结果
│   └── v1_baseline_summary.json       # 摘要
│
└── plots/                             # 图表目录
    └── v1_baseline/                   # 按实验 ID 组织
        ├── v1_baseline_hits.png
        ├── v1_baseline_mrr.png
        ├── v1_baseline_recall.png
        ├── v1_baseline_hit_by_category.png
        ├── v1_baseline_latency_distribution.png
        ├── v1_baseline_latency_boxplot.png
        └── v1_baseline_answer_quality.png
```

---

## 🏷️ 版本命名规则

**完整实验 ID** = `{experiment_version}_{experiment_name}`

| experiment_version | experiment_name | 完整 ID |
|-------------------|-----------------|---------|
| `v1` | `baseline` | `v1_baseline` |
| `v2` | `with_bm25` | `v2_with_bm25` |
| `None` | `quick_test` | `quick_test` |

---

## 📊 自动生成的图表（7 张）

| # | 图表 | 文件名后缀 |
|---|------|-----------|
| 1 | 命中率分布 | `_hits.png` |
| 2 | MRR 分布 | `_mrr.png` |
| 3 | 召回率分布 | `_recall.png` |
| 4 | 分类命中率 | `_hit_by_category.png` |
| 5 | 延迟分布 | `_latency_distribution.png` |
| 6 | 延迟箱线图 | `_latency_boxplot.png` |
| 7 | 答案质量 | `_answer_quality.png` |

---

## ⚙️ 配置选项

```python
EvaluationConfig(
    # 检索指标
    k_values=[1, 3, 5, 10, 20],
    
    # RAGAS
    ragas_enabled=True,
    
    # WandB
    wandb_enabled=False,
    wandb_project="cuad-assistant",
    
    # 输出
    output_dir="results/evaluation",
    
    # 绘图
    plot_enabled=True,                    # 是否生成图表
    plot_dir="results/plots",             # 图表目录
    plot_include_performance=True,        # 延迟图
    plot_include_quality=True,            # 答案质量图
    plot_include_correlation=False,       # 相关性热图
)
```

---

## 🔄 多版本对比实验示例

```bash
# 版本 1: Vanilla
python scripts/run_vanilla_rag.py \
  --mode test \
  --experiment-name "vanilla" \
  --max-samples 100

# 版本 2: 添加 BM25
python scripts/run_evaluation.py \
  --mode test \
  --experiment-name "with_bm25" \
  --experiment-version "v2"

# 版本 3: 添加 Reranker
python scripts/run_evaluation.py \
  --mode test \
  --experiment-name "with_rerank" \
  --experiment-version "v3"
```

**结果目录**:
```
results/plots/
├── vanilla_vanilla/
├── v2_with_bm25/
└── v3_with_rerank/
```

---

## 📋 控制台输出示例

```
Experiment ID: v1_baseline

Evaluating: 100%|████████████| 100/100 [01:30<00:00, 1.1it/s]

==================================================
EVALUATION SUMMARY
==================================================

--- Retrieval Metrics ---
  @ 1: Hit=0.1230, MRR=0.1230, Recall=0.1450
  @ 5: Hit=0.3650, MRR=0.2340, Recall=0.4210
  @10: Hit=0.4520, MRR=0.2890, Recall=0.5210

--- Answer Metrics ---
  f1          : 0.3120
  exact_match : 0.0850

--- Latency Metrics ---

  RETRIEVAL:
    Mean:     125.3 ms
    P50:      118.0 ms
    P90:      175.0 ms

  GENERATION:
    Mean:    1850.2 ms
    P50:    1720.0 ms
    P90:    2450.0 ms

==================================================

--------------------------------------------------
OUTPUT FILES
--------------------------------------------------
  csv         : results/evaluation/v1_baseline.csv
  summary     : results/evaluation/v1_baseline_summary.json
  plots_dir   : results/plots/v1_baseline
--------------------------------------------------
```

---

## 🎉 快速开始

```bash
# 最简用法
python scripts/run_vanilla_rag.py --mode test --max-samples 10

# 查看输出
ls results/evaluation/
ls results/plots/
```

**所有评估结果和图表将自动生成和版本化！** 📊✨
