"""
WandB 实验追踪器

功能:
- 记录实验参数
- 记录评估指标
- 记录图表和表格
- 支持实验对比

使用方式:
    from src.evaluation.wandb_tracker import WandBTracker
    
    tracker = WandBTracker(project="cuad-assistant", experiment_name="v1.0")
    tracker.log_config(config)
    tracker.log_metrics(metrics)
    tracker.finish()
"""

import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from loguru import logger

# WandB 导入
try:
    import wandb
    from wandb import Artifact
    WANDB_AVAILABLE = True
except ImportError:
    logger.warning("WandB not available")
    WANDB_AVAILABLE = False


@dataclass
class WandBConfig:
    """WandB 配置"""
    
    enabled: bool = True
    project: str = "cuad-assistant"
    entity: str = ""  # 组织/用户名
    experiment_name: str = ""
    tags: List[str] = None
    notes: str = ""
    
    # 高级选项
    mode: str = "online"  # online, offline, disabled
    log_model: bool = True
    log_code: bool = True
    
    @classmethod
    def from_config(cls, config) -> 'WandBConfig':
        """从全局配置创建"""
        try:
            wandb_cfg = config.evaluation.wandb
            return cls(
                enabled=getattr(wandb_cfg, 'enabled', True),
                project=getattr(wandb_cfg, 'project', 'cuad-assistant'),
                entity=getattr(wandb_cfg, 'entity', ''),
            )
        except Exception:
            return cls()


class WandBTracker:
    """
    WandB 实验追踪器
    
    用于记录和追踪 RAG 系统的实验:
    - 配置参数
    - 评估指标
    - 可视化图表
    - 模型 artifacts
    """
    
    def __init__(
        self,
        config: Optional[WandBConfig] = None,
        project: str = None,
        experiment_name: str = None,
        tags: List[str] = None,
        **kwargs,
    ):
        """
        初始化 WandB 追踪器
        
        Args:
            config: WandB 配置
            project: 项目名称 (覆盖 config)
            experiment_name: 实验名称
            tags: 标签列表
        """
        self.config = config or WandBConfig()
        
        # 覆盖配置
        if project:
            self.config.project = project
        if experiment_name:
            self.config.experiment_name = experiment_name
        if tags:
            self.config.tags = tags
        
        self._run = None
        self._initialized = False
    
    def init(
        self,
        config_dict: Dict[str, Any] = None,
        resume: bool = False,
        **kwargs,
    ) -> bool:
        """
        初始化 WandB run
        
        Args:
            config_dict: 实验配置字典
            resume: 是否恢复之前的 run
        
        Returns:
            是否成功初始化
        """
        if not WANDB_AVAILABLE:
            logger.warning("WandB not installed, tracking disabled")
            return False
        
        if not self.config.enabled:
            logger.info("WandB tracking disabled in config")
            return False
        
        try:
            # 生成实验名称
            name = self.config.experiment_name
            if not name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"experiment_{timestamp}"
            
            # 初始化 run
            self._run = wandb.init(
                project=self.config.project,
                entity=self.config.entity or None,
                name=name,
                tags=self.config.tags,
                notes=self.config.notes,
                config=config_dict,
                resume=resume,
                mode=self.config.mode,
                **kwargs,
            )
            
            self._initialized = True
            logger.success(f"WandB initialized: {self._run.url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            return False
    
    @property
    def is_active(self) -> bool:
        """检查是否已初始化"""
        return self._initialized and self._run is not None
    
    def log_config(self, config: Union[Dict[str, Any], Any]) -> None:
        """
        记录配置
        
        Args:
            config: 配置字典或配置对象
        """
        if not self.is_active:
            return
        
        # 转换为字典
        if hasattr(config, '__dict__'):
            config_dict = self._dataclass_to_dict(config)
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config
        
        wandb.config.update(config_dict)
        logger.debug(f"Logged config to WandB")
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int = None,
        prefix: str = "",
    ) -> None:
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 步骤数
            prefix: 指标前缀
        """
        if not self.is_active:
            return
        
        # 添加前缀
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # 过滤非数值和特殊字段
        filtered = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not k.startswith('_')
        }
        
        if step is not None:
            wandb.log(filtered, step=step)
        else:
            wandb.log(filtered)
        
        logger.debug(f"Logged {len(filtered)} metrics to WandB")
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """
        记录摘要指标 (run 结束时的最终值)
        
        Args:
            summary: 摘要字典
        """
        if not self.is_active:
            return
        
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                wandb.run.summary[k] = v
        
        logger.debug(f"Logged summary to WandB")
    
    def log_table(
        self,
        name: str,
        data: List[Dict[str, Any]],
        columns: List[str] = None,
    ) -> None:
        """
        记录表格
        
        Args:
            name: 表格名称
            data: 数据列表
            columns: 列名列表
        """
        if not self.is_active:
            return
        
        if not data:
            return
        
        if columns is None:
            columns = list(data[0].keys())
        
        # 创建 WandB Table
        table = wandb.Table(columns=columns)
        for row in data:
            table.add_data(*[row.get(c) for c in columns])
        
        wandb.log({name: table})
        logger.debug(f"Logged table '{name}' with {len(data)} rows")
    
    def log_chart(
        self,
        title: str,
        data: Dict[str, List],
        chart_type: str = "line",
        x_key: str = None,
        y_keys: List[str] = None,
    ) -> None:
        """
        记录图表
        
        Args:
            title: 图表标题
            data: 数据字典
            chart_type: 图表类型 (line, bar, scatter)
            x_key: X 轴数据的 key
            y_keys: Y 轴数据的 keys
        """
        if not self.is_active:
            return
        
        # 使用 wandb.plot
        if chart_type == "line":
            chart = wandb.plot.line_series(
                xs=data.get(x_key, list(range(len(next(iter(data.values())))))),
                ys=[data[k] for k in (y_keys or data.keys())],
                keys=y_keys or list(data.keys()),
                title=title,
            )
        elif chart_type == "bar":
            table = wandb.Table(data=[[k, v] for k, v in data.items()], columns=["label", "value"])
            chart = wandb.plot.bar(table, "label", "value", title=title)
        else:
            logger.warning(f"Unsupported chart type: {chart_type}")
            return
        
        wandb.log({title: chart})
    
    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """
        记录 artifact (模型/数据/配置)
        
        Args:
            name: artifact 名称
            artifact_type: 类型 (model, dataset, config)
            path: 文件/目录路径
            metadata: 元数据
        """
        if not self.is_active:
            return
        
        artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
        
        path = Path(path)
        if path.is_file():
            artifact.add_file(str(path))
        elif path.is_dir():
            artifact.add_dir(str(path))
        else:
            logger.warning(f"Artifact path not found: {path}")
            return
        
        wandb.log_artifact(artifact)
        logger.info(f"Logged artifact: {name} ({artifact_type})")
    
    def log_evaluation_results(
        self,
        retrieval_metrics: Dict[str, float] = None,
        answer_metrics: Dict[str, float] = None,
        ragas_metrics: Dict[str, float] = None,
        detailed_results: List[Dict[str, Any]] = None,
        step: int = None,
    ) -> None:
        """
        记录完整的评估结果
        
        Args:
            retrieval_metrics: 检索指标
            answer_metrics: 答案指标
            ragas_metrics: RAGAS 指标
            detailed_results: 详细结果
            step: 步骤数
        """
        if not self.is_active:
            return
        
        # 记录各类指标
        if retrieval_metrics:
            self.log_metrics(retrieval_metrics, step=step, prefix="retrieval")
        
        if answer_metrics:
            self.log_metrics(answer_metrics, step=step, prefix="answer")
        
        if ragas_metrics:
            self.log_metrics(ragas_metrics, step=step, prefix="ragas")
        
        # 记录详细结果表格
        if detailed_results:
            self.log_table("evaluation_details", detailed_results[:100])
    
    def finish(self) -> None:
        """结束 WandB run"""
        if self._run:
            self._run.finish()
            self._run = None
            self._initialized = False
            logger.info("WandB run finished")
    
    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """将 dataclass 转换为字典"""
        from dataclasses import fields, is_dataclass
        
        if not is_dataclass(obj):
            return obj
        
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if is_dataclass(value):
                result[f.name] = self._dataclass_to_dict(value)
            elif isinstance(value, list):
                result[f.name] = [
                    self._dataclass_to_dict(v) if is_dataclass(v) else v
                    for v in value
                ]
            elif isinstance(value, dict):
                result[f.name] = {
                    k: self._dataclass_to_dict(v) if is_dataclass(v) else v
                    for k, v in value.items()
                }
            else:
                result[f.name] = value
        
        return result
    
    def __enter__(self):
        """上下文管理器入口"""
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.finish()


# =============================================================================
# 便捷函数
# =============================================================================

_global_tracker: Optional[WandBTracker] = None


def init_wandb(
    project: str = "cuad-assistant",
    experiment_name: str = None,
    config: Dict[str, Any] = None,
    **kwargs,
) -> WandBTracker:
    """
    初始化全局 WandB 追踪器
    
    Args:
        project: 项目名称
        experiment_name: 实验名称
        config: 配置字典
    
    Returns:
        WandBTracker 实例
    """
    global _global_tracker
    
    _global_tracker = WandBTracker(
        project=project,
        experiment_name=experiment_name,
        **kwargs,
    )
    _global_tracker.init(config_dict=config)
    
    return _global_tracker


def get_tracker() -> Optional[WandBTracker]:
    """获取全局追踪器"""
    return _global_tracker


def log_metrics(metrics: Dict[str, Any], **kwargs) -> None:
    """记录指标到全局追踪器"""
    if _global_tracker and _global_tracker.is_active:
        _global_tracker.log_metrics(metrics, **kwargs)


def finish_wandb() -> None:
    """结束全局追踪器"""
    global _global_tracker
    if _global_tracker:
        _global_tracker.finish()
        _global_tracker = None
