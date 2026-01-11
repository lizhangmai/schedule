"""
Cluster 类：管理多个 GPU 资源
"""

from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .gpu import GPU
    from .task import Task


# 从配置文件导入 GPU 配置
try:
    from config.gpu_configs import GPU_CONFIGS
except ImportError:
    # 如果导入失败，使用默认值（需与 config/gpu_configs.py 保持一致）
    GPU_CONFIGS = {
        "A100": {"memory_capacity": 80, "scaling_factor": 57.0},
        "A30": {"memory_capacity": 24, "scaling_factor": 30.0},
        "L40": {"memory_capacity": 48, "scaling_factor": 55.5},
    }


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


def create_small_cluster() -> Cluster:
    """创建小规模集群：3 个 GPU (A100, A30, L40 各 1 个)"""
    configs = [
        {"gpu_id": "A100-1", "model": "A100",
         "memory_capacity": GPU_CONFIGS["A100"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["A100"]["scaling_factor"]},
        {"gpu_id": "A30-1", "model": "A30",
         "memory_capacity": GPU_CONFIGS["A30"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["A30"]["scaling_factor"]},
        {"gpu_id": "L40-1", "model": "L40",
         "memory_capacity": GPU_CONFIGS["L40"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["L40"]["scaling_factor"]},
    ]
    return Cluster.from_configs(configs)


def create_medium_cluster() -> Cluster:
    """创建中等规模集群：6 个 GPU (每种 2 个)"""
    configs = [
        {"gpu_id": "A100-1", "model": "A100",
         "memory_capacity": GPU_CONFIGS["A100"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["A100"]["scaling_factor"]},
        {"gpu_id": "A100-2", "model": "A100",
         "memory_capacity": GPU_CONFIGS["A100"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["A100"]["scaling_factor"]},
        {"gpu_id": "A30-1", "model": "A30",
         "memory_capacity": GPU_CONFIGS["A30"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["A30"]["scaling_factor"]},
        {"gpu_id": "A30-2", "model": "A30",
         "memory_capacity": GPU_CONFIGS["A30"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["A30"]["scaling_factor"]},
        {"gpu_id": "L40-1", "model": "L40",
         "memory_capacity": GPU_CONFIGS["L40"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["L40"]["scaling_factor"]},
        {"gpu_id": "L40-2", "model": "L40",
         "memory_capacity": GPU_CONFIGS["L40"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["L40"]["scaling_factor"]},
    ]
    return Cluster.from_configs(configs)


def create_large_cluster() -> Cluster:
    """创建大规模集群：9 个 GPU (每种 3 个)"""
    configs = [
        {"gpu_id": f"A100-{i}", "model": "A100",
         "memory_capacity": GPU_CONFIGS["A100"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["A100"]["scaling_factor"]}
        for i in range(1, 4)
    ] + [
        {"gpu_id": f"A30-{i}", "model": "A30",
         "memory_capacity": GPU_CONFIGS["A30"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["A30"]["scaling_factor"]}
        for i in range(1, 4)
    ] + [
        {"gpu_id": f"L40-{i}", "model": "L40",
         "memory_capacity": GPU_CONFIGS["L40"]["memory_capacity"],
         "scaling_factor": GPU_CONFIGS["L40"]["scaling_factor"]}
        for i in range(1, 4)
    ]
    return Cluster.from_configs(configs)
