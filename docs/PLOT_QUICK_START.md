# 绘图功能快速开始 🎨

**1 分钟了解新功能**

---

## ✨ 核心功能

```bash
# 运行测试 → 自动生成 7 张图表
python scripts/run_vanilla_rag.py --mode test --max-samples 10
```

---

## 📊 自动生成的图表

| # | 图表类型 | 文件名 | 用途 |
|---|---------|--------|------|
| 1 | **命中率** | `_hits.png` | 检索成功率 |
| 2 | **MRR** | `_mrr.png` | 排名质量 |
| 3 | **召回率** | `_recall.png` | 覆盖率 |
| 4 | **分类性能** | `_hit_by_category.png` | 各条款类型对比 |
| 5 | **延迟分布** | `_latency_distribution.png` | 性能统计 |
| 6 | **延迟箱线图** | `_latency_boxplot.png` | 性能对比 |
| 7 | **答案质量** | `_answer_quality.png` | F1/Precision/Recall |

---

## 🚀 常用命令

### 基础测试（带图表）
```bash
python scripts/run_vanilla_rag.py --mode test --max-samples 10
```

### 自定义实验名
```bash
python scripts/run_vanilla_rag.py \
  --mode test \
  --experiment-name "my_experiment"
```

### 禁用图表（更快）
```bash
python scripts/run_vanilla_rag.py --mode test --no-plots
```

### 完整测试
```bash
python scripts/run_vanilla_rag.py --mode test
```

---

## 📁 输出位置

```
results/
├── vanilla_rag/
│   └── vanilla_rag_results.csv          # 数据
└── plots/
    └── vanilla_rag_20240115_143022/     # 图表目录
        ├── ..._hits.png
        ├── ..._mrr.png
        ├── ..._recall.png
        ├── ..._hit_by_category.png
        ├── ..._latency_distribution.png
        ├── ..._latency_boxplot.png
        └── ..._answer_quality.png
```

---

## 🔍 查看图表

### macOS
```bash
open results/plots/vanilla_rag_*/
```

### Linux
```bash
xdg-open results/plots/vanilla_rag_*/*.png
```

### Windows
```bash
explorer results\plots\vanilla_rag_*\
```

---

## 📊 图表示例

### 命中率分布
```
Miss: 245 (48.0%)  ████████
Hit:  265 (52.0%)  █████████
```

### 延迟统计
```
Retrieval:  Mean 125ms, P95 180ms
Generation: Mean 1.8s,  P95 2.5s
Total:      Mean 2.3s,  P95 3.5s
```

### 答案质量
```
F1 Score:     0.312 ± 0.298
Exact Match:  8.5%
```

---

## 🎯 版本对比

```bash
# 运行多个版本
python scripts/run_vanilla_rag.py --experiment-name "v1_baseline"
python scripts/run_vanilla_rag.py --experiment-name "v2_optimized"

# 对比图表
python -c "
from src.utils.plot_enhanced import plot_version_comparison
import pandas as pd

results = {
    'Baseline': pd.read_csv('results/vanilla_rag/vanilla_rag_results.csv'),
    'Optimized': pd.read_csv('results/vanilla_rag/vanilla_rag_results.csv'),
}

plot_version_comparison(results, ['hit@10', 'f1_score'])
"
```

---

## ⚙️ 安装依赖

```bash
pip install matplotlib seaborn scipy
```

或

```bash
pip install -r requirements.txt
```

---

## 📖 完整文档

- 详细说明: `docs/PLOT_INTEGRATION_GUIDE.md`
- 系统就绪: `docs/SYSTEM_READINESS_CHECK.md`
- 输出说明: `docs/VANILLA_RAG_OUTPUT_GUIDE.md`

---