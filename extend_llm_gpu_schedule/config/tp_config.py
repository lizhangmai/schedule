"""
Tensor parallelism configuration for LLM inference.

Defines TP allocation strategies and recommended TP degrees
for different (model, GPU) combinations.
"""

from typing import Dict, List, Optional, Set
from .llm_model_specs import LLM_MODEL_SPECS, get_model_spec


# Tensor parallelism allocation strategies
TP_ALLOCATION_STRATEGIES = {
    "power_of_2": [1, 2, 4, 8, 16],  # Traditional power-of-2 TP
    "flexible": [1, 2, 3, 4, 6, 8, 12, 16],  # More flexible for odd GPU counts
}


# Recommended TP configurations per model and GPU
# Based on memory requirements and performance considerations
RECOMMENDED_TP_CONFIG: Dict[str, Dict[str, int]] = {
    "DeepSeek-R1": {
        "H100": 8,      # 671B / 8 = 84GB per GPU (fits in H100 96GB)
        "H100-96GB": 8, # 84GB per GPU
        "A100": 8,      # 84GB per GPU (fits in A100 80GB)
        "A100-40GB": 8, # 84GB per GPU (doesn't fit, but max TP)
        "A30": 4,       # 168GB per GPU (doesn't fit, max TP)
        "L40": 4,       # 168GB per GPU (doesn't fit, max TP)
    },
    "Qwen3": {
        "H100": 2,      # 15GB per GPU (comfortable fit)
        "H100-96GB": 2,
        "A100": 2,      # 15GB per GPU
        "A100-40GB": 1, # 30GB single GPU feasible
        "A30": 1,       # 30GB single GPU feasible
        "L40": 1,       # 30GB single GPU feasible
    },
    "GLM4.7": {
        "H100": 4,      # 89GB per GPU (fits in H100 96GB, tight in 80GB)
        "H100-96GB": 4, # 89GB per GPU (comfortable fit)
        "A100": 4,      # 89GB per GPU (tight fit in 80GB)
        "A100-40GB": 4, # 89GB per GPU (doesn't fit, max TP)
        "A30": 2,       # 178GB per GPU (doesn't fit, max TP)
        "L40": 2,       # 178GB per GPU (doesn't fit, max TP)
    },
    "Llama3-70B": {
        "H100": 4,      # 35GB per GPU (comfortable fit)
        "H100-96GB": 4,
        "A100": 4,      # 35GB per GPU
        "A100-40GB": 2, # 70GB per GPU
        "A30": 1,       # 140GB (doesn't fit in 24GB, needs TP=2+)
        "L40": 2,       # 70GB per GPU
    },
    "Llama3-8B": {
        "H100": 1,      # 16GB single GPU
        "H100-96GB": 1,
        "A100": 1,
        "A100-40GB": 1,
        "A30": 1,       # 16GB fits
        "L40": 1,
    },
    "Mistral-7B": {
        "H100": 1,      # 14GB single GPU
        "H100-96GB": 1,
        "A100": 1,
        "A100-40GB": 1,
        "A30": 1,       # 14GB fits
        "L40": 1,
    },
    "Mixtral-8x7B": {
        "H100": 2,      # 47GB per GPU
        "H100-96GB": 2,
        "A100": 2,      # 47GB per GPU
        "A100-40GB": 2, # 47GB per GPU
        "A30": 1,       # 94GB (doesn't fit, needs TP=2+)
        "L40": 2,       # 47GB per GPU
    },
}


def get_recommended_tp(model_name: str, gpu_model: str) -> Optional[int]:
    """
    Get recommended tensor parallelism degree for a model on a GPU.

    Args:
        model_name: LLM model name
        gpu_model: GPU model name

    Returns:
        Recommended TP degree, or None if model not found
    """
    if model_name not in RECOMMENDED_TP_CONFIG:
        return None
    return RECOMMENDED_TP_CONFIG[model_name].get(gpu_model, 2)


def is_tp_supported(model_name: str, tp_degree: int) -> bool:
    """
    Check if TP degree is supported for a model.

    Args:
        model_name: LLM model name
        tp_degree: Tensor parallelism degree to check

    Returns:
        True if TP degree is within model's valid range
    """
    spec = get_model_spec(model_name)
    if spec is None:
        return False
    return spec.validate_tp_degree(tp_degree)


def can_fit_on_gpu(model_name: str, gpu_model: str, tp_degree: int, gpu_memory_gb: float) -> bool:
    """
    Check if model can fit on GPU with given TP degree.

    Args:
        model_name: LLM model name
        gpu_model: GPU model name
        tp_degree: Tensor parallelism degree
        gpu_memory_gb: GPU memory capacity in GB

    Returns:
        True if model fits on the GPU
    """
    spec = get_model_spec(model_name)
    if spec is None:
        return False

    mem_per_gpu = spec.get_memory_for_tp(tp_degree)
    if mem_per_gpu is None:
        return False

    return mem_per_gpu <= gpu_memory_gb


def find_min_tp_for_gpu(model_name: str, gpu_memory_gb: float) -> Optional[int]:
    """
    Find minimum TP degree for model to fit on GPU.

    Args:
        model_name: LLM model name
        gpu_memory_gb: GPU memory capacity in GB

    Returns:
        Minimum TP degree that fits, or None if doesn't fit at max TP
    """
    spec = get_model_spec(model_name)
    if spec is None:
        return None

    # Try from min to max TP
    for tp_degree in range(spec.min_tp, spec.max_tp + 1):
        mem_per_gpu = spec.get_memory_for_tp(tp_degree)
        if mem_per_gpu is not None and mem_per_gpu <= gpu_memory_gb:
            return tp_degree

    return None


def get_valid_tp_degrees(model_name: str, gpu_memory_gb: float) -> List[int]:
    """
    Get all valid TP degrees for model on GPU.

    Args:
        model_name: LLM model name
        gpu_memory_gb: GPU memory capacity in GB

    Returns:
        List of valid TP degrees (sorted)
    """
    spec = get_model_spec(model_name)
    if spec is None:
        return []

    valid_tps = []
    for tp_degree in range(spec.min_tp, spec.max_tp + 1):
        if can_fit_on_gpu(model_name, "", tp_degree, gpu_memory_gb):
            valid_tps.append(tp_degree)

    return valid_tps


def optimize_tp_allocation(
    model_names: List[str],
    gpu_models: List[str],
    gpu_memory_gb: float
) -> Dict[str, int]:
    """
    Optimize TP allocation for multiple models on available GPUs.

    Simple greedy approach: assign minimum TP that fits for each model.

    Args:
        model_names: List of model names to allocate
        gpu_models: List of available GPU models
        gpu_memory_gb: GPU memory capacity (assumed uniform)

    Returns:
        Dictionary mapping model_name -> TP degree
    """
    allocation = {}

    for model_name in model_names:
        min_tp = find_min_tp_for_gpu(model_name, gpu_memory_gb)
        if min_tp is not None:
            allocation[model_name] = min_tp

    return allocation


def calculate_tp_scaling_efficiency(
    model_name: str,
    tp_degree: int,
    phase: str = "decode"
) -> float:
    """
    Estimate multi-GPU scaling efficiency for TP.

    Decode phase typically has lower scaling due to communication overhead.

    Args:
        model_name: LLM model name
        tp_degree: Tensor parallelism degree
        phase: "prefill" or "decode"

    Returns:
        Scaling efficiency (0.0 to 1.0)
    """
    if tp_degree <= 1:
        return 1.0

    # Base efficiency decreases with TP degree
    # Prefill (compute-bound) scales better than decode (memory-bound)
    if phase.lower() == "prefill":
        # Compute-bound: better scaling
        base_efficiency = 0.95
        scaling_penalty = 0.02 * (tp_degree - 1)
    else:
        # Memory-bound: worse scaling due to communication
        base_efficiency = 0.85
        scaling_penalty = 0.05 * (tp_degree - 1)

    efficiency = max(0.5, base_efficiency - scaling_penalty)
    return efficiency


# GPU memory capacities for reference
GPU_MEMORY_CAPACITIES = {
    "H100": 80.0,
    "H100-96GB": 96.0,
    "A100": 80.0,
    "A100-40GB": 40.0,
    "A30": 24.0,
    "L40": 48.0,
    "L4": 24.0,
}


if __name__ == "__main__":
    # Print recommended TP configurations
    print("Recommended Tensor Parallelism Configurations:")
    print("-" * 70)
    for model, gpu_configs in RECOMMENDED_TP_CONFIG.items():
        print(f"\n{model}:")
        for gpu, tp in gpu_configs.items():
            print(f"  {gpu:15} -> TP{tp}")
    print("-" * 70)

    # Test fitting on different GPUs
    print("\nModel Fitting on Different GPUs:")
    print("=" * 70)
    for gpu_model, gpu_mem in [("H100", 80.0), ("A100", 80.0), ("A30", 24.0)]:
        print(f"\n{gpu_model} ({gpu_mem}GB):")
        for model_name in ["DeepSeek-R1", "Qwen3", "GLM4.7", "Llama3-8B"]:
            min_tp = find_min_tp_for_gpu(model_name, gpu_mem)
            rec_tp = get_recommended_tp(model_name, gpu_model)
            if min_tp:
                mem = get_model_spec(model_name).get_memory_for_tp(min_tp)
                fit_status = "✓" if mem <= gpu_mem else "✗"
                print(f"  {model_name:15} -> Min TP: {min_tp}, Rec TP: {rec_tp}, "
                      f"Mem: {mem}GB {fit_status}")
            else:
                print(f"  {model_name:15} -> Doesn't fit ✗")
