"""
GPU 配置定义
"""

from typing import Dict, List

# GPU 型号配置
# scaling_factor 基于：
# 1. 实际 GPU 算力比值 (A100 FP32: ~19.5 TFLOPS, A30: ~10.3 TFLOPS, L40: ~19.1 TFLOPS)
# 2. 任务时间要求 - 让单个 GPU 无法独立完成所有任务，必须多 GPU 协作
BASE_SCALE = 40

GPU_CONFIGS: Dict[str, Dict[str, int | float]] = {
    "A100": {
        "memory_capacity": 80,  # GB
        "scaling_factor": BASE_SCALE * 1.9,  # ≈ 30 × 1.9 (A100 约为 A30 的 1.9 倍)
    },
    "A30": {
        "memory_capacity": 24,  # GB
        "scaling_factor": BASE_SCALE,  # 基准值，99.9% 的任务可按时完成
    },
    "L40": {
        "memory_capacity": 48,  # GB
        "scaling_factor": BASE_SCALE * 1.85,  # ≈ 30 × 1.85 (L40 约为 A30 的 1.85 倍)
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
