# Reranker模型显存和性能优化分析

## 问题分析

### 1. 为什么reranker模型会占用23G显存？

#### 主要原因

**a) 激活值（Activations）显存占用**
- 虽然模型权重使用4bit量化（约5G），但推理时的激活值仍然是float16/bfloat16
- 激活值显存计算公式：
  ```
  激活值显存 ≈ batch_size × seq_len × hidden_size × num_layers × 2 (float16) × 中间层缓存
  ```
- 对于Qwen3-Reranker-4B模型：
  - hidden_size ≈ 4096
  - num_layers ≈ 32
  - 当前配置：batch_size=32, max_length=512
  - 激活值显存 ≈ 32 × 512 × 4096 × 32 × 2 × 2 ≈ **8-12GB**
  
**b) 模型权重显存**
- 4bit量化后模型权重：约5G
- 但量化模型在推理时仍需要一些额外的显存用于：
  - 量化参数（scale/zero_point）
  - 反量化缓存
  - 总计约：**6-7GB**

**c) KV Cache和其他开销**
- Attention机制的KV cache
- 中间计算结果缓存
- PyTorch框架开销
- 总计约：**2-4GB**

**总计：6-7GB（权重）+ 8-12GB（激活值）+ 2-4GB（其他）≈ 16-23GB**

#### 耗时长的原因

1. **批处理大小不合适**：batch_size=32对于长序列来说太大，导致：
   - 显存占用高
   - 计算时间长
   - 可能触发显存交换

2. **序列长度**：max_length=512仍然较长，虽然已从1024降低

3. **缺少优化**：
   - 未使用`torch.compile()`进行图优化
   - 未使用`torch.inference_mode()`替代`torch.no_grad()`
   - 数据传输未使用异步（non_blocking）

4. **内存管理**：
   - GPU缓存清理不够及时
   - 可能存在显存碎片化

### 2. 如何优化处理速度？

#### 优化策略

**a) 减小批处理大小**
- 将batch_size从32降低到4-8
- 虽然会增加批次数，但可以：
  - 显著降低显存占用
  - 提高单批处理速度
  - 避免OOM错误

**b) 进一步减少序列长度**
- 将max_length从512降低到256-384
- 对于大多数文档片段，256-384已经足够

**c) 使用torch.inference_mode()**
- 比`torch.no_grad()`更快，禁用更多梯度相关计算

**d) 异步数据传输**
- 使用`non_blocking=True`进行GPU数据传输
- 可以重叠计算和数据传输

**e) 定期清理GPU缓存**
- 每处理几个batch后清理一次，避免显存碎片化

**f) 使用torch.compile()（可选）**
- PyTorch 2.0+支持，可以显著加速推理
- 但需要额外编译时间

### 3. 还有什么使用了GPU1的显存？

#### GPU1显存占用分析

**a) Embedding模型（all-MiniLM-L6-v2）**
- 模型大小：约80MB（权重）
- 推理时激活值：约50-100MB
- **总计约：150-200MB**

**b) Reranker模型（Qwen3-Reranker-4B，4bit量化）**
- 模型权重（4bit量化）：约5G
- 量化参数和缓存：约1-2G
- **总计约：6-7GB**

**c) Reranker推理时的激活值**
- batch_size=32, max_length=512时：约8-12GB
- batch_size=8, max_length=256时：约2-3GB

**d) 显存碎片化**
- PyTorch内存分配器可能产生碎片
- 可能占用额外1-2GB

**总计（stage 1阶段，batch_size=32时）：**
- Embedding: 0.2GB
- Reranker权重: 6-7GB
- Reranker激活值: 8-12GB
- 碎片化: 1-2GB
- **总计：15-21GB**（接近23GB显存上限）

## 优化方案

### 方案1：减小批处理大小和序列长度（推荐）

**优点**：
- 立即生效，无需修改模型
- 显著降低显存占用
- 提高单批处理速度

**缺点**：
- 批次数增加，总时间可能略增（但单批更快）

**实施**：
- batch_size: 32 → 4-8
- max_length: 512 → 256-384

### 方案2：使用更高效的推理模式

**优点**：
- 提升推理速度
- 降低显存占用

**实施**：
- 使用`torch.inference_mode()`替代`torch.no_grad()`
- 使用`non_blocking=True`进行数据传输
- 定期清理GPU缓存

### 方案3：使用torch.compile()（可选）

**优点**：
- 显著加速推理（20-30%提升）

**缺点**：
- 需要PyTorch 2.0+
- 首次编译需要时间
- 可能增加显存占用

### 方案4：延迟加载Embedding模型

**优点**：
- 减少初始显存占用

**实施**：
- 在需要时才加载Embedding模型
- 或者将Embedding模型移到CPU（如果速度可接受）

## 推荐配置

### 保守配置（确保不OOM）
```python
RERANKER_MAX_LENGTH = 256
batch_size = 4
```

### 平衡配置（推荐）
```python
RERANKER_MAX_LENGTH = 384
batch_size = 8
```

### 激进配置（如果显存充足）
```python
RERANKER_MAX_LENGTH = 512
batch_size = 16
```

## 实施建议

1. **立即实施**：减小batch_size到4-8，max_length到256-384
2. **优化代码**：使用torch.inference_mode()和non_blocking传输
3. **监控显存**：使用nvidia-smi监控实际显存使用
4. **逐步调整**：根据实际显存情况逐步增加batch_size
