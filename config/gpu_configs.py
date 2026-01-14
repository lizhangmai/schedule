"""
GPU 配置定义
"""

from typing import Dict, List

# GPU 型号配置
# scaling_factor 基于：
# 1. 实际 GPU 算力比值 (A100 FP32: ~19.5 TFLOPS, A30: ~10.3 TFLOPS, L40: ~19.1 TFLOPS)
# 2. 任务时间要求 - 让单个 GPU 无法独立完成所有任务，必须多 GPU 协作
DEFAULT_BASE_SCALE = 30

GPU_CONFIGS: Dict[str, Dict[str, int | float]] = {
    "A100": {
        "memory_capacity": 80,  # GB
        "scaling_factor": DEFAULT_BASE_SCALE * 1.9,  # ≈ 30 × 1.9 (A100 约为 A30 的 1.9 倍)
    },
    "A30": {
        "memory_capacity": 24,  # GB
        "scaling_factor": DEFAULT_BASE_SCALE,  # 基准值，99.9% 的任务可按时完成
    },
    "L40": {
        "memory_capacity": 48,  # GB
        "scaling_factor": DEFAULT_BASE_SCALE * 1.85,  # ≈ 30 × 1.85 (L40 约为 A30 的 1.85 倍)
    },
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
        "A100": {
            "memory_capacity": 80,
            "scaling_factor": base_scale * 1.9,
        },
        "A30": {
            "memory_capacity": 24,
            "scaling_factor": base_scale,
        },
        "L40": {
            "memory_capacity": 48,
            "scaling_factor": base_scale * 1.85,
        },
    }


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

    configs = {
        "small": [
            {"gpu_id": "A100-1", "model": "A100", **gpu_configs["A100"]},
            {"gpu_id": "A30-1", "model": "A30", **gpu_configs["A30"]},
            {"gpu_id": "L40-1", "model": "L40", **gpu_configs["L40"]},
        ],
        "medium": [
            {"gpu_id": "A100-1", "model": "A100", **gpu_configs["A100"]},
            {"gpu_id": "A100-2", "model": "A100", **gpu_configs["A100"]},
            {"gpu_id": "A30-1", "model": "A30", **gpu_configs["A30"]},
            {"gpu_id": "A30-2", "model": "A30", **gpu_configs["A30"]},
            {"gpu_id": "L40-1", "model": "L40", **gpu_configs["L40"]},
            {"gpu_id": "L40-2", "model": "L40", **gpu_configs["L40"]},
        ],
        "large": [
            {"gpu_id": f"A100-{i}", "model": "A100", **gpu_configs["A100"]} for i in range(1, 4)
        ]
        + [
            {"gpu_id": f"A30-{i}", "model": "A30", **gpu_configs["A30"]} for i in range(1, 4)
        ]
        + [
            {"gpu_id": f"L40-{i}", "model": "L40", **gpu_configs["L40"]} for i in range(1, 4)
        ],
    }

    return configs.get(size, configs["small"])

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
