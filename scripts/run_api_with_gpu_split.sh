#!/bin/bash
# GPU分配策略启动脚本 - API服务
# LLM推理使用GPU0，其他模块（embedding、reranker）使用GPU1

# 设置CUDA_VISIBLE_DEVICES，让所有GPU可见
export CUDA_VISIBLE_DEVICES=0,1

echo "启动RAG API服务..."
echo "GPU分配策略："
echo "  - LLM推理 (vLLM): GPU0"
echo "  - Embedding模型: GPU1"
echo "  - Reranker模型: GPU1"
echo ""

# 运行API服务
cd "$(dirname "$0")/.."
python scripts/run_api.py
