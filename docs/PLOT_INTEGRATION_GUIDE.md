# Vanilla RAG 可视化集成指南

**新功能**: 自动生成评估图表，支持版本管理和对比分析

---

## 📊 新增功能概览

### ✅ 已集成到测试流程

| 功能 | 说明 |
|------|------|
| **自动绘图** | 测试完成后自动生成所有图表 |
| **版本管理** | 每次实验独立目录，方便对比 |
| **丰富图表** | 10+ 种可视化图表 |
| **性能分析** | 延迟分布、箱线图 |
| **质量分析** | F1、Precision、Recall 分布 |

---

## 🎨 生成的图表类型

### 1. 检索指标图表（4 张）

| 图表 | 文件名 | 说明 |
|------|--------|------|
| **命中率分布** | `{experiment}_hits.png` | Hit@10 的命中/未命中统计 |
| **MRR 分布** | `{experiment}_mrr.png` | 平均倒数排名分布 |
| **召回率分布** | `{experiment}_recall.png` | Recall@10 值分布 |
| **分类命中率** | `{experiment}_hit_by_category.png` | 各条款类型的命中率对比 |

### 2. 性能指标图表（2 张）

| 图表 | 文件名 | 说明 |
|------|--------|------|
| **延迟分布** | `{experiment}_latency_distribution.png` | 检索/生成/总时间的分布和统计 |
| **延迟箱线图** | `{experiment}_latency_boxplot.png` | 三个时间指标的箱线图对比 |

### 3. 答案质量图表（1 张）

| 图表 | 文件名 | 说明 |
|------|--------|------|
| **质量指标** | `{experiment}_answer_quality.png` | F1/Precision/Recall/EM 分布 |

### 总计: **7 张核心图表**

---

## 🚀 使用方式

### 基础用法（自动生成图表）

```bash
# 默认：自动生成图表，自动命名
python scripts/run_vanilla_rag.py --mode test --max-samples 10
```

**输出结构**:
```
results/
├── vanilla_rag/
│   ├── vanilla_rag_results.csv
│   └── vanilla_rag_results_summary.json
└── plots/
    └── vanilla_rag_20240115_143022/    # 自动时间戳
        ├── vanilla_rag_20240115_143022_hits.png
        ├── vanilla_rag_20240115_143022_mrr.png
        ├── vanilla_rag_20240115_143022_recall.png
        ├── vanilla_rag_20240115_143022_hit_by_category.png
        ├── vanilla_rag_20240115_143022_latency_distribution.png
        ├── vanilla_rag_20240115_143022_latency_boxplot.png
        └── vanilla_rag_20240115_143022_answer_quality.png
```

---

### 自定义实验名称

```bash
# 指定实验名称（方便后续查找和对比）
python scripts/run_vanilla_rag.py \
  --mode test \
  --max-samples 100 \
  --experiment-name "v1_baseline"
```

**输出结构**:
```
results/plots/
└── v1_baseline/
    ├── v1_baseline_hits.png
    ├── v1_baseline_mrr.png
    └── ...
```

---

### 禁用图表生成

```bash
# 仅需要 CSV 数据，不生成图表（加快测试速度）
python scripts/run_vanilla_rag.py \
  --mode test \
  --no-plots
```

---

## 📈 图表示例说明

### 1. 命中率分布 (Hits)

```
┌─────────────────────────────┐
│  Hit Distribution (hit@10)  │
├─────────────────────────────┤
│                             │
│   █████                     │
│   █████  245 (48.0%)        │
│   █████                     │
│   █████                     │
│   █████  ███                │
│   █████  ███  265 (52.0%)  │
│   Miss   Hit                │
└─────────────────────────────┘
```

**解读**:
- 红色（Miss）：未命中的样本数和比例
- 绿色（Hit）：命中的样本数和比例

---

### 2. MRR 分布

```
┌─────────────────────────────┐
│   MRR Distribution          │
├─────────────────────────────┤
│ 最高分布在 0.0 和 1.0        │
│ 体现检索排名质量             │
└─────────────────────────────┘
```

**解读**:
- 0.0：完全未命中
- 1.0：首位命中
- 0.33：第三位命中
- 0.5：第二位命中

---

### 3. 延迟分布 (Latency Distribution)

```
┌──────────────────────────────────────────────┐
│  Retrieval Time   Generation Time   Total    │
├──────────────────────────────────────────────┤
│  直方图 + KDE 曲线                            │
│  Mean: 125ms      Mean: 1850ms    Mean: 2.3s│
│  Median: 120ms    Median: 1800ms  Median:2.2s│
│  P95: 180ms       P95: 2500ms     P95: 3.5s │
└──────────────────────────────────────────────┘
```

**解读**:
- 绿线（Mean）：平均值
- 橙线（Median）：中位数
- 红线（P95）：95 分位数

---

### 4. 延迟箱线图 (Latency Boxplot)

```
┌─────────────────────────────┐
│  Latency Comparison         │
├─────────────────────────────┤
│      ╭───┬───╮              │
│      │   │   │              │
│  ────┴───┴───┴────          │
│   Ret  Gen  Total           │
└─────────────────────────────┘
```

**解读**:
- 箱体：25%-75% 分位数
- 中线：中位数
- 须：最小/最大值（排除异常值）

---

### 5. 答案质量指标 (Answer Quality)

```
┌────────────────────────────────────────┐
│  F1 Score      Precision               │
│  Mean: 0.312   Mean: 0.345            │
│                                        │
│  Recall        Exact Match            │
│  Mean: 0.298   Rate: 8.5%            │
└────────────────────────────────────────┘
```

---

### 6. 分类命中率 (Hit by Category)

```
┌─────────────────────────────┐
│  Hit@10 by Category         │
├─────────────────────────────┤
│  Termination  ████████ 0.53│
│  Payment      ██████   0.45│
│  Liability    █████    0.38│
│  ...                        │
└─────────────────────────────┘
```

**解读**: 横向柱状图，颜色越绿表示命中率越高

---

## 🔄 版本对比实验

### 场景：对比 Vanilla vs Enhanced 版本

```bash
# 1. 运行 Vanilla 版本
python scripts/run_vanilla_rag.py \
  --mode test \
  --experiment-name "v1_vanilla"

# 2. 运行 Enhanced 版本（添加 BM25）
python scripts/run_enhanced_rag.py \
  --mode test \
  --experiment-name "v2_with_bm25"

# 3. 运行 Enhanced 版本（添加 Reranker）
python scripts/run_enhanced_rag.py \
  --mode test \
  --use-rerank \
  --experiment-name "v3_with_rerank"
```

**输出结构**:
```
results/plots/
├── v1_vanilla/          # 基线版本
│   ├── ...
├── v2_with_bm25/        # BM25 增强
│   ├── ...
└── v3_with_rerank/      # Reranker 增强
    ├── ...
```

---

### 生成对比图

使用 Python 脚本对比多个版本：

```python
from src.utils.plot_enhanced import plot_version_comparison
import pandas as pd

# 加载多个版本的结果
results = {
    'Vanilla': pd.read_csv('results/vanilla_rag/v1_vanilla.csv'),
    'BM25': pd.read_csv('results/vanilla_rag/v2_with_bm25.csv'),
    'Rerank': pd.read_csv('results/vanilla_rag/v3_with_rerank.csv'),
}

# 对比关键指标
plot_version_comparison(
    results,
    metrics=['hit@10', 'mrr@10', 'f1_score', 'total_time_ms'],
    output_dir='results/plots/comparison'
)
```

**输出**: `results/plots/comparison/version_comparison.png`

---

## 📊 图表详细配置

### 当前配置

| 参数 | 值 | 说明 |
|------|-----|------|
| **DPI** | 300 | 高清晰度 |
| **图表尺寸** | 8-12 英寸 | 适合报告 |
| **颜色方案** | Seaborn Husl | 色盲友好 |
| **字体大小** | 10-14pt | 清晰可读 |

### 自定义绘图

如果需要自定义图表，可以直接使用绘图工具：

```python
from src.utils.plot_enhanced import PlotManager, plot_all_metrics
import pandas as pd

# 读取结果
df = pd.read_csv('results/vanilla_rag/vanilla_rag_results.csv')

# 生成所有图表（含相关性热图）
plot_all_metrics(
    df,
    loc='my_experiment',
    include_performance=True,
    include_quality=True,
    include_correlation=True,  # 启用相关性热图
)
```

---

## 🎯 典型工作流

### 1. 快速验证（无图表）

```bash
# 10 样本快速测试，无图表
python scripts/run_vanilla_rag.py \
  --mode test \
  --max-samples 10 \
  --no-plots
```

**耗时**: ~1 分钟

---

### 2. 完整基线测试（含图表）

```bash
# 所有样本，生成完整图表
python scripts/run_vanilla_rag.py \
  --mode test \
  --experiment-name "baseline_20240115"
```

**输出**:
- CSV 数据
- JSON 摘要
- 7 张图表

**耗时**: ~25-30 分钟

---

### 3. 多版本对比实验

```bash
# 基线
python scripts/run_vanilla_rag.py \
  --mode test \
  --experiment-name "exp1_vanilla"

# 变体 1
python scripts/run_vanilla_rag.py \
  --mode test \
  --top-k 20 \
  --experiment-name "exp2_topk20"

# 变体 2
python scripts/run_vanilla_rag.py \
  --mode test \
  --top-k 5 \
  --experiment-name "exp3_topk5"
```

**对比分析**:
```bash
# 使用 Jupyter Notebook 或 Python 脚本
# 加载所有实验的 CSV 数据
# 生成对比图表
```

---

## 📂 文件组织结构

```
results/
├── vanilla_rag/                    # CSV 数据目录
│   ├── vanilla_rag_results.csv
│   └── vanilla_rag_results_summary.json
│
├── plots/                          # 图表目录
│   ├── baseline_20240115/          # 实验 1
│   │   ├── baseline_20240115_hits.png
│   │   ├── baseline_20240115_mrr.png
│   │   └── ...
│   ├── exp_bm25_20240116/          # 实验 2
│   │   └── ...
│   └── comparison/                 # 对比图
│       └── version_comparison.png
│
└── reports/                        # 报告（可选）
    └── analysis_20240115.md
```

---

## 🛠️ 故障排除

### 问题 1: 绘图失败但评估成功

**现象**:
```
✅ Results saved to: results/vanilla_rag/...
⚠️  Failed to generate plots: ...
```

**解决方案**:
```bash
# 手动生成图表
python -c "
from src.utils.plot_enhanced import plot_all_metrics
import pandas as pd
df = pd.read_csv('results/vanilla_rag/vanilla_rag_results.csv')
plot_all_metrics(df, 'manual_plot')
"
```

---

### 问题 2: 缺少依赖

**现象**: `ModuleNotFoundError: No module named 'seaborn'`

**解决方案**:
```bash
pip install seaborn scipy
```

---

### 问题 3: 图表显示异常

**现象**: 中文乱码、字体警告

**解决方案**:
```python
# 在绘图前添加
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

---

## 📊 性能考虑

| 样本数 | 评估耗时 | 绘图耗时 | 总耗时 |
|--------|---------|---------|--------|
| 10 | ~30s | ~5s | ~35s |
| 100 | ~5min | ~8s | ~5.5min |
| 500 | ~25min | ~12s | ~25.5min |

**结论**: 绘图开销很小（< 1% 总时间），建议默认开启

---

## 🎨 高级功能

### 1. 自定义绘图配置

```python
# 创建自定义绘图管理器
from src.utils.plot_enhanced import PlotManager

manager = PlotManager(base_dir='results/custom_plots')
exp_dir = manager.get_experiment_dir('my_experiment')
print(f"Plots will be saved to: {exp_dir}")
```

---

### 2. 批量重新生成图表

```bash
# 为所有历史实验重新生成图表
python -c "
import pandas as pd
from pathlib import Path
from src.utils.plot_enhanced import plot_all_metrics

csv_dir = Path('results/vanilla_rag')
for csv_file in csv_dir.glob('*.csv'):
    df = pd.read_csv(csv_file)
    exp_name = csv_file.stem
    plot_all_metrics(df, exp_name)
    print(f'✅ Regenerated plots for: {exp_name}')
"
```

---

### 3. 导出论文级图表

```python
# 高质量导出（论文、报告用）
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600  # 超高清
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (12, 8)

# 然后运行绘图
from src.utils.plot_enhanced import plot_all_metrics
plot_all_metrics(df, 'paper_quality')
```

---

## 总结

### ✅ 自动化流程

```
运行测试
    ↓
生成 CSV 数据
    ↓
自动计算指标
    ↓
自动生成 7 张图表
    ↓
保存到版本化目录
```

### 🎯 下一步

1. **运行第一个测试**:
   ```bash
   python scripts/run_vanilla_rag.py --mode test --max-samples 10
   ```

2. **查看生成的图表**:
   ```bash
   ls results/plots/vanilla_rag_*/
   ```

3. **对比多个版本**:
   - 运行多个实验
   - 使用 `plot_version_comparison()` 生成对比图

---

**准备好了吗？开始你的第一个带图表的评估！** 📊🚀
