"""
Cluster 类：管理多个 GPU 资源
"""

# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .gpu import GPU
    from .task import Task

# 从配置文件导入 GPU 配置
from config.gpu_configs import (
    DEFAULT_BASE_SCALE,
    get_cluster_config as get_cluster_config_from_file,
)


@dataclass
class Cluster:
    """
    集群类：管理多个 GPU 资源

    属性:
        gpus: List[GPU] - GPU 列表
    """

    gpus: List["GPU"] = field(default_factory=list)

    def __post_init__(self):
        """初始化后构建 GPU 索引"""
        self._gpu_index: Dict[str, "GPU"] = {gpu.gpu_id: gpu for gpu in self.gpus}

    @classmethod
    def from_configs(cls, configs: List[Dict]) -> "Cluster":
        """
        从配置列表创建集群

        Args:
            configs: GPU 配置列表，每个配置包含：
                - gpu_id: GPU ID
                - model: GPU 型号
                - memory_capacity: 显存容量
                - scaling_factor: 计算能力因子

        Returns:
            Cluster 对象
        """
        from .gpu import GPU

        gpus = [GPU(**config) for config in configs]
        return cls(gpus=gpus)

    def get_gpu(self, gpu_id: str) -> Optional["GPU"]:
        """
        根据 ID 获取 GPU

        Args:
            gpu_id: GPU ID

        Returns:
            GPU 对象，如果不存在返回 None
        """
        return self._gpu_index.get(gpu_id)

    def get_available_gpus(self, task: "Task") -> List["GPU"]:
        """
        获取能够容纳该任务的 GPU 列表

        Args:
            task: 任务对象

        Returns:
            可以容纳任务的 GPU 列表
        """
        return [gpu for gpu in self.gpus if gpu.can_accommodate(task)]

    def get_total_memory_capacity(self) -> int:
        """获取集群总显存容量"""
        return sum(gpu.memory_capacity for gpu in self.gpus)

    def get_gpu_count(self) -> int:
        """获取 GPU 数量"""
        return len(self.gpus)

    def get_cluster_statistics(self) -> Dict:
        """
        获取集群整体统计信息

        Returns:
            统计信息字典
        """
        return {
            "gpu_count": len(self.gpus),
            "total_memory_capacity": self.get_total_memory_capacity(),
            "gpu_models": {gpu.gpu_id: gpu.model for gpu in self.gpus},
        }

    def reset(self) -> None:
        """重置集群状态（清空所有 GPU 时间线和任务调度状态）"""
        for gpu in self.gpus:
            # 重置所有关联任务的调度状态
            for _, _, task in gpu.timeline:
                task.reset()
            # 清空时间线
            gpu.timeline.clear()

    def __repr__(self) -> str:
        gpu_info = ", ".join(str(gpu) for gpu in self.gpus)
        return f"Cluster([{gpu_info}])"


def create_cluster(size: str, scaling_factor: float = DEFAULT_BASE_SCALE) -> Cluster:
    """
    创建指定规模的集群

    Args:
        size: 集群规模 (small/medium/large)
        scaling_factor: 基准缩放因子（A30 的 scaling_factor）

    Returns:
        集群对象
    """
    configs = get_cluster_config_from_file(size, scaling_factor)
    return Cluster.from_configs(configs)


def create_small_cluster(scaling_factor: float = DEFAULT_BASE_SCALE) -> Cluster:
    """
    创建小规模集群：3 个 GPU (A100, A30, L40 各 1 个)

    Args:
        scaling_factor: 基准缩放因子（A30 的 scaling_factor）

    Returns:
        小规模集群对象
    """
    return create_cluster("small", scaling_factor)


def create_medium_cluster(scaling_factor: float = DEFAULT_BASE_SCALE) -> Cluster:
    """
    创建中等规模集群：6 个 GPU (每种 2 个)

    Args:
        scaling_factor: 基准缩放因子（A30 的 scaling_factor）

    Returns:
        中等规模集群对象
    """
    return create_cluster("medium", scaling_factor)


def create_large_cluster(scaling_factor: float = DEFAULT_BASE_SCALE) -> Cluster:
    """
    创建大规模集群：9 个 GPU (每种 3 个)

    Args:
        scaling_factor: 基准缩放因子（A30 的 scaling_factor）

    Returns:
        大规模集群对象
    """
    return create_cluster("large", scaling_factor)
