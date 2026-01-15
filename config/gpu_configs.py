"""
GPU 配置定义
"""

from typing import Dict, List

# GPU 型号配置
# scaling_factor 基于：
# 1. 实际 GPU 算力比值 (A100 FP32: ~19.5 TFLOPS, A30: ~10.3 TFLOPS, L40: ~19.1 TFLOPS)
# 2. 任务时间要求 - 让单个 GPU 无法独立完成所有任务，必须多 GPU 协作
DEFAULT_BASE_SCALE = 30

# GPU 相对性能因子（相对于 A30 的倍数）
GPU_SCALING_RATIOS = {
    "A100": 1.9,
    "A30": 1.0,
    "L40": 1.85,
}

# GPU 显存容量（GB）
GPU_MEMORY_CAPACITIES = {
    "A100": 80,
    "A30": 24,
    "L40": 48,
}


def get_gpu_configs_with_scale(base_scale: float = DEFAULT_BASE_SCALE) -> Dict[str, Dict[str, int | float]]:
    """
    获取指定基准缩放因子的 GPU 配置

    Args:
        base_scale: 基准缩放因子（A30 的 scaling_factor）

    Returns:
        GPU 配置字典
    """
    return {
        model: {
            "memory_capacity": GPU_MEMORY_CAPACITIES[model],
            "scaling_factor": base_scale * GPU_SCALING_RATIOS[model],
        }
        for model in GPU_SCALING_RATIOS
    }


def _create_gpu_configs_list(gpu_configs: Dict[str, Dict], size: str) -> List[Dict]:
    """
    根据集群规模创建 GPU 配置列表

    Args:
        gpu_configs: GPU 配置字典
        size: 集群规模 (small/medium/large)

    Returns:
        GPU 配置列表
    """
    size_counts = {
        "small": 1,
        "medium": 2,
        "large": 3,
    }

    count = size_counts.get(size, 1)
    configs = []

    for model in ["A100", "A30", "L40"]:
        for i in range(1, count + 1):
            configs.append({
                "gpu_id": f"{model}-{i}",
                "model": model,
                **gpu_configs[model],
            })

    return configs


def get_cluster_config(size: str, base_scale: float = DEFAULT_BASE_SCALE) -> List[Dict]:
    """
    获取指定规模的集群配置

    Args:
        size: 集群规模 (small/medium/large)
        base_scale: 基准缩放因子（A30 的 scaling_factor）

    Returns:
        GPU 配置列表
    """
    gpu_configs = get_gpu_configs_with_scale(base_scale)
    return _create_gpu_configs_list(gpu_configs, size)


# 集群配置（不同规模）- 使用默认缩放因子
GPU_CONFIGS = get_gpu_configs_with_scale(DEFAULT_BASE_SCALE)
CLUSTER_CONFIGS: Dict[str, List[Dict]] = {
    size: _create_gpu_configs_list(GPU_CONFIGS, size)
    for size in ["small", "medium", "large"]
}


def get_gpu_config(model: str, base_scale: float = DEFAULT_BASE_SCALE) -> Dict[str, int | float]:
    """
    获取指定 GPU 型号的配置

    Args:
        model: GPU 型号 (A100/A30/L40)
        base_scale: 基准缩放因子（A30 的 scaling_factor）

    Returns:
        GPU 配置字典
    """
    configs = get_gpu_configs_with_scale(base_scale)
    return configs.get(model, {})
