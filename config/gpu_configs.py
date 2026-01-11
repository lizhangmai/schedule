"""
GPU 配置定义
"""

from typing import Dict, List

# GPU 型号配置
GPU_CONFIGS: Dict[str, Dict[str, int | float]] = {
    "A100": {
        "memory_capacity": 80,  # GB
        "scaling_factor": 2.0,
    },
    "A30": {
        "memory_capacity": 24,  # GB
        "scaling_factor": 1.0,
    },
    "L40": {
        "memory_capacity": 48,  # GB
        "scaling_factor": 1.5,
    },
}

# 集群配置（不同规模）
CLUSTER_CONFIGS: Dict[str, List[Dict]] = {
    "small": [
        {"gpu_id": "A100-1", "model": "A100", **GPU_CONFIGS["A100"]},
        {"gpu_id": "A30-1", "model": "A30", **GPU_CONFIGS["A30"]},
        {"gpu_id": "L40-1", "model": "L40", **GPU_CONFIGS["L40"]},
    ],
    "medium": [
        {"gpu_id": "A100-1", "model": "A100", **GPU_CONFIGS["A100"]},
        {"gpu_id": "A100-2", "model": "A100", **GPU_CONFIGS["A100"]},
        {"gpu_id": "A30-1", "model": "A30", **GPU_CONFIGS["A30"]},
        {"gpu_id": "A30-2", "model": "A30", **GPU_CONFIGS["A30"]},
        {"gpu_id": "L40-1", "model": "L40", **GPU_CONFIGS["L40"]},
        {"gpu_id": "L40-2", "model": "L40", **GPU_CONFIGS["L40"]},
    ],
    "large": [
        {"gpu_id": f"A100-{i}", "model": "A100", **GPU_CONFIGS["A100"]} for i in range(1, 4)
    ]
    + [
        {"gpu_id": f"A30-{i}", "model": "A30", **GPU_CONFIGS["A30"]} for i in range(1, 4)
    ]
    + [
        {"gpu_id": f"L40-{i}", "model": "L40", **GPU_CONFIGS["L40"]} for i in range(1, 4)
    ],
}


def get_gpu_config(model: str) -> Dict[str, int | float]:
    """获取指定 GPU 型号的配置"""
    return GPU_CONFIGS.get(model, {})


def get_cluster_config(size: str) -> List[Dict]:
    """获取指定规模的集群配置"""
    return CLUSTER_CONFIGS.get(size, CLUSTER_CONFIGS["small"])
