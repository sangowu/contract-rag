#!/usr/bin/env python
"""
API 服务启动脚本

使用方式:
    # 开发模式 (热重载)
    python scripts/run_api.py --mode dev
    
    # 测试模式
    python scripts/run_api.py --mode test
    
    # 生产模式
    python scripts/run_api.py --mode prod
    
    # 指定端口
    python scripts/run_api.py --mode dev --port 8080
    
    # 指定 workers
    python scripts/run_api.py --mode prod --workers 4
    
    # 强制单 GPU 模式 (即使有多 GPU)
    python scripts/run_api.py --mode dev --gpu-mode single_gpu
    
    # 强制双 GPU 模式
    python scripts/run_api.py --mode prod --gpu-mode dual_gpu

GPU 模式说明:
    - single_gpu: 使用 Qwen3-4B 模型，所有服务共享一个 GPU
    - dual_gpu: 使用 Qwen3-8B 模型，LLM 独占 GPU0，其他服务使用 GPU1
    - auto (默认): 根据检测到的 GPU 数量自动选择
"""

import os
import sys
import argparse
from loguru import logger

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Start CUAD API Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='prod',
        choices=['test', 'dev', 'prod'],
        help='Environment mode (default: prod)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Port to bind (default: 8000)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of workers (default: 1 for dev, 4 for prod)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload (default: True for dev)'
    )
    
    parser.add_argument(
        '--no-reload',
        action='store_true',
        help='Disable auto-reload'
    )
    
    parser.add_argument(
        '--gpu-mode',
        type=str,
        default=None,
        choices=['single_gpu', 'dual_gpu'],
        help='Force GPU mode: single_gpu (4B model) or dual_gpu (8B model). Default: auto-detect'
    )
    
    parser.add_argument(
        '--show-gpu-info',
        action='store_true',
        help='Show GPU information and exit'
    )
    
    return parser.parse_args()


def show_gpu_info():
    """显示 GPU 信息"""
    import torch
    
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    print()
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024 ** 3)
        print(f"GPU {i}: {props.name}")
        print(f"  - Total Memory: {memory_gb:.2f} GB")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - Multi-Processor Count: {props.multi_processor_count}")
        print()
    
    # 推荐模式
    if gpu_count == 1:
        print("Recommended mode: single_gpu (Qwen3-4B model)")
    else:
        print("Recommended mode: dual_gpu (Qwen3-8B model)")
    
    print("=" * 60)


def main():
    """主函数"""
    import uvicorn
    
    args = parse_args()
    
    # 显示 GPU 信息并退出
    if args.show_gpu_info:
        show_gpu_info()
        return
    
    # 设置环境变量
    os.environ['APP_MODE'] = args.mode
    
    # 设置强制 GPU 模式
    if args.gpu_mode:
        os.environ['FORCE_GPU_MODE'] = args.gpu_mode
    
    # 根据模式设置默认值
    if args.mode == 'dev':
        workers = args.workers or 1
        reload = not args.no_reload
    elif args.mode == 'test':
        workers = args.workers or 1
        reload = not args.no_reload
    else:  # prod
        workers = args.workers or 4
        reload = args.reload and not args.no_reload
    
    # 显示启动信息
    logger.info("=" * 60)
    logger.info("Starting CUAD API Service")
    logger.info("=" * 60)
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Host: {args.host}:{args.port}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Reload: {reload}")
    logger.info(f"  GPU Mode: {args.gpu_mode or 'auto'}")
    logger.info("=" * 60)
    
    # 启动服务
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
