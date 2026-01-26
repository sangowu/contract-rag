# GPU分配策略说明

## 概述

为了避免GPU内存不足（OOM）问题，系统采用双GPU分配策略：

- **GPU0**: 专门用于LLM推理（vLLM）
- **GPU1**: 用于其他模块（Embedding模型、Reranker模型）

## 配置说明

### 1. GPU设备分配常量

在 `src/utils/model_loading.py` 中定义了GPU分配常量：

```python
LLM_GPU_ID = 0  # LLM推理使用GPU0
OTHER_GPU_ID = 1  # Embedding和Reranker使用GPU1
```

### 2. 模块GPU分配

#### LLM推理 (vLLM)
- **位置**: `src/inference/llm_inference.py`
- **GPU**: GPU0
- **实现**: 通过 `get_vllm()` 函数，如果未设置 `CUDA_VISIBLE_DEVICES`，会自动设置为GPU0
- **注意**: vLLM通过环境变量 `CUDA_VISIBLE_DEVICES` 控制可见GPU

#### Embedding模型
- **位置**: `src/rag/embedding.py`
- **GPU**: GPU1
- **实现**: `get_model()` 函数自动检测GPU1可用性，如果可用则使用GPU1，否则降级到CPU

#### Reranker模型
- **位置**: `src/rag/retrieval.py` -> `load_reranker()`
- **GPU**: GPU1
- **实现**: 通过 `device_map` 参数明确指定GPU1

## 使用方法

### 方法1: 使用启动脚本（推荐）

#### 运行评估脚本
```bash
./scripts/run_evaluate_with_gpu_split.sh
```

#### 运行API服务
```bash
./scripts/run_api_with_gpu_split.sh
```

### 方法2: 手动设置环境变量

在运行Python脚本前设置环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0,1
python scripts/evaluate_script.py
```

或者：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/evaluate_script.py
```

## 注意事项

1. **环境变量设置时机**: 
   - `CUDA_VISIBLE_DEVICES` 必须在导入任何CUDA相关库之前设置
   - 建议在启动脚本中设置，而不是在Python代码中设置

2. **单GPU环境**:
   - 如果系统只有单GPU，代码会自动降级
   - Embedding和Reranker会尝试使用GPU0（如果GPU1不可用）

3. **vLLM的特殊性**:
   - vLLM在初始化时会自动选择可见的GPU
   - 如果 `CUDA_VISIBLE_DEVICES=0,1`，vLLM默认会使用GPU0
   - 为了确保vLLM只使用GPU0，可以在启动时设置 `CUDA_VISIBLE_DEVICES=0`，但这会隐藏GPU1

4. **内存管理**:
   - LLM的GPU内存利用率默认设置为0.4（可在代码中调整）
   - 如果仍然遇到OOM，可以进一步降低 `gpu_util` 参数

## 验证GPU分配

运行以下命令检查GPU使用情况：

```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

然后运行你的脚本，观察：
- GPU0应该显示vLLM进程
- GPU1应该显示embedding/reranker相关进程

## 故障排除

### 问题1: vLLM仍然使用所有GPU

**解决方案**: 确保在启动Python进程前设置 `CUDA_VISIBLE_DEVICES=0`（仅对vLLM进程），或者使用子进程隔离。

### 问题2: Embedding/Reranker无法使用GPU1

**检查**:
1. 确认系统有多个GPU: `nvidia-smi`
2. 检查 `torch.cuda.device_count()` 返回值
3. 查看日志中的GPU分配信息

### 问题3: 仍然出现OOM错误

**解决方案**:
1. 进一步降低 `gpu_util` 参数（例如从0.4降到0.3）
2. 检查是否有其他进程占用GPU内存
3. 考虑使用量化模型或更小的模型

## 代码修改位置总结

1. **`src/utils/model_loading.py`**:
   - 添加GPU分配常量
   - 修改 `get_vllm()` 支持GPU0
   - 修改 `load_reranker()` 支持GPU1

2. **`src/rag/embedding.py`**:
   - 修改 `get_model()` 使用GPU1

3. **`src/rag/retrieval.py`**:
   - 修改 `rerank_results()` 正确处理设备分配

4. **启动脚本**:
   - `scripts/run_evaluate_with_gpu_split.sh`
   - `scripts/run_api_with_gpu_split.sh`
