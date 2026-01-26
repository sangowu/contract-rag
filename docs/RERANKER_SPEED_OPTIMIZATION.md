# Reranker速度优化说明

## 优化目标

利用剩余10G显存，进一步提升reranker的处理速度。

## 已实施的优化

### 1. 增大batch_size（关键优化）

**从256增加到512**
- **原因**：还有10G显存可用，可以支持更大的batch_size
- **效果**：减少批次数，提高GPU利用率
- **显存占用**：batch_size=512, max_length=256时，激活值约4-6GB（仍在安全范围）

### 2. 智能tokenize策略

**自适应tokenize方式**：
- **小数据集（≤10000条）**：预先tokenize所有pairs，减少循环开销
- **大数据集（>10000条）**：分批tokenize，避免CPU内存压力

**优势**：
- 对于常见场景（通常<1000条），预先tokenize可以显著减少循环开销
- 对于超大数据集，自动降级到分批tokenize，避免OOM

### 3. 减少CPU-GPU数据传输

**优化前**：
```python
scores.extend(batch_relevance_scores.cpu().tolist())  # 每个batch都转换
```

**优化后**：
```python
all_scores.append(batch_relevance_scores)  # 保持在GPU上
# 最后统一转换
all_scores_tensor = torch.cat(all_scores, dim=0)
scores = all_scores_tensor.cpu().tolist()
```

**优势**：
- 减少CPU-GPU数据传输次数
- 利用GPU的并行计算能力进行合并
- 减少Python循环开销

### 4. 使用torch.topk替代Python sort

**优化前**：
```python
scored_candidates.sort(key=lambda x: x[0], reverse=True)  # Python排序
```

**优化后**：
```python
top_k_values, top_k_indices = torch.topk(all_scores_tensor, k=top_k, largest=True)  # GPU排序
```

**优势**：
- torch.topk在GPU上执行，比Python sort快得多
- 对于top_k场景，只需要找到前k个，不需要完整排序

### 5. 优化内存管理

**减少清理频率**：
- 从每8个batch清理一次改为每16个batch清理一次
- 因为batch_size更大，清理频率可以降低

**及时释放变量**：
- 在循环中立即释放不需要的中间变量
- 对于分批tokenize的情况，立即释放batch_inputs

## 性能提升预期

### 显存占用

**优化前（batch_size=256）**：
- 权重：6-7GB
- 激活值：4-5GB
- 其他：1-2GB
- **总计：11-14GB**

**优化后（batch_size=512）**：
- 权重：6-7GB
- 激活值：4-6GB
- 其他：1-2GB
- **总计：11-15GB**（仍在23GB安全范围内）

### 处理速度

**预期提升**：
- **批次数减少**：约50%（batch_size从256到512）
- **tokenize开销减少**：对于小数据集，约30-50%减少
- **排序速度提升**：使用torch.topk，约2-5倍提升
- **总体速度提升**：约30-60%（取决于数据量）

### 适用场景

**最佳场景**：
- candidate_chunks数量：100-5000条
- 显存充足：剩余10G+显存
- 需要top_k排序

**注意事项**：
- 如果candidate_chunks数量>10000，会自动降级到分批tokenize
- 如果显存不足，可以降低batch_size

## 进一步优化建议（可选）

### 1. 动态batch_size调整

根据实际显存使用情况动态调整batch_size：
```python
# 检测可用显存
free_memory = torch.cuda.get_device_properties(model_device).total_memory - torch.cuda.memory_allocated(model_device)
# 根据可用显存调整batch_size
optimal_batch_size = calculate_optimal_batch_size(free_memory, max_length)
```

### 2. 使用torch.jit.script（如果支持）

对于某些操作，可以使用torch.jit.script进行加速（但4bit量化模型可能不支持）。

### 3. 多流处理（如果支持）

使用CUDA streams进行异步处理，重叠计算和数据传输。

## 使用建议

1. **监控显存使用**：使用`nvidia-smi`监控实际显存占用
2. **根据实际情况调整**：如果出现OOM，降低batch_size
3. **测试不同配置**：根据实际数据量测试最优batch_size

## 代码变更总结

1. `batch_size`默认值：256 → 512
2. 智能tokenize策略：根据数据量自动选择
3. GPU上合并scores：减少CPU-GPU传输
4. torch.topk排序：替代Python sort
5. 优化内存管理：减少清理频率，及时释放变量
