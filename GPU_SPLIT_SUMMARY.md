# GPU分配策略实施总结

## 问题描述

原始代码中所有模块可能共享同一GPU，导致GPU内存不足（OOM）错误。错误信息显示：
```
ValueError: Free memory on device (1.14/23.52 GiB) on startup is less than desired GPU memory utilization (0.8, 18.81 GiB)
```

## 解决方案

实施双GPU分配策略：
- **GPU0**: 专门用于LLM推理（vLLM）
- **GPU1**: 用于其他模块（Embedding模型、Reranker模型）

## 修改的文件

### 1. `src/utils/model_loading.py`
- ✅ 添加GPU分配常量 `LLM_GPU_ID = 0` 和 `OTHER_GPU_ID = 1`
- ✅ 修改 `get_vllm()` 函数，支持固定在GPU0
- ✅ 修改 `load_reranker()` 函数，明确指定使用GPU1
- ✅ 添加详细的日志记录，便于调试

### 2. `src/rag/embedding.py`
- ✅ 修改 `get_model()` 函数，自动检测并使用GPU1
- ✅ 如果GPU1不可用，自动降级到CPU

### 3. `src/rag/retrieval.py`
- ✅ 修改 `rerank_results()` 函数，正确处理量化模型的设备分配
- ✅ 确保输入数据被正确移动到GPU1

### 4. 新增文件
- ✅ `scripts/run_evaluate_with_gpu_split.sh` - 评估脚本启动器
- ✅ `scripts/run_api_with_gpu_split.sh` - API服务启动器
- ✅ `docs/GPU_ALLOCATION.md` - 详细的GPU分配说明文档

## 使用方法

### 快速开始

#### 运行评估（推荐使用启动脚本）
```bash
./scripts/run_evaluate_with_gpu_split.sh
```

#### 运行API服务
```bash
./scripts/run_api_with_gpu_split.sh
```

#### 手动设置环境变量
```bash
export CUDA_VISIBLE_DEVICES=0,1
python scripts/evaluate_script.py
```

## 技术细节

### vLLM GPU分配
- vLLM通过 `CUDA_VISIBLE_DEVICES` 环境变量控制可见GPU
- 如果未设置，代码会自动尝试设置为GPU0
- **建议**: 在启动脚本中明确设置 `CUDA_VISIBLE_DEVICES=0,1`

### HuggingFace模型GPU分配
- Embedding和Reranker模型通过 `device_map` 参数或 `.to()` 方法指定GPU1
- 对于量化模型，使用 `device_map` 参数更可靠

### 降级策略
- 如果系统只有单GPU，代码会自动降级
- 所有模块会尝试使用可用的GPU
- 如果GPU不可用，会自动使用CPU

## 验证

系统已确认有2个GPU可用：
```
CUDA available: True
GPU count: 2
```

## 预期效果

1. **内存隔离**: LLM推理和其他模块使用不同的GPU，避免内存竞争
2. **性能提升**: 并行处理，提高整体吞吐量
3. **稳定性**: 减少OOM错误的发生

## 注意事项

1. **环境变量设置时机**: `CUDA_VISIBLE_DEVICES` 必须在导入CUDA库之前设置
2. **vLLM特殊性**: vLLM在初始化时会自动选择可见的GPU，需要特别注意
3. **内存管理**: 如果仍然遇到OOM，可以进一步降低 `gpu_util` 参数（默认0.4）

## 后续优化建议

1. 如果仍然遇到内存问题，考虑：
   - 进一步降低 `gpu_util` 参数
   - 使用量化模型
   - 减少batch size

2. 监控GPU使用情况：
   ```bash
   watch -n 1 nvidia-smi
   ```

3. 根据实际使用情况调整GPU分配策略

## 相关文档

- 详细说明: `docs/GPU_ALLOCATION.md`
- 代码修改: 见上述文件列表
