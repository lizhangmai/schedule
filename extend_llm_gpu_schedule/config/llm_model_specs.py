"""
LLM model specifications for scheduling.

Defines model parameters, memory requirements, and tensor parallelism
configurations for popular LLMs including DeepSeek-R1, Qwen3, GLM4.7.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class LLMModelSpec:
    """
    LLM model specification for scheduling decisions.

    Attributes:
        name: Model name (e.g., "DeepSeek-R1", "Qwen3")
        parameters: Model parameters in billions
        memory_per_gpu: Memory required per GPU for different TP degrees (GB)
        min_tp: Minimum tensor parallelism degree
        max_tp: Maximum tensor parallelism degree
        recommended_tp: Recommended tensor parallelism configuration
        flops_per_token_prefill: FLOPs required per token in prefill phase
        flops_per_token_decode: FLOPs required per token in decode phase
        kv_cache_per_token: KV cache memory per token in GB (approximate)
        description: Model description
    """
    name: str
    parameters: float  # Billions
    memory_per_gpu: Dict[int, float]  # TP degree -> memory per GPU (GB)
    min_tp: int
    max_tp: int
    recommended_tp: int
    flops_per_token_prefill: float  # FLOPs per token
    flops_per_token_decode: float  # FLOPs per token
    kv_cache_per_token: float  # GB per token (approximate)
    description: str

    def get_memory_for_tp(self, tp_degree: int) -> Optional[float]:
        """
        Get memory per GPU for a given tensor parallelism degree.

        Args:
            tp_degree: Tensor parallelism degree

        Returns:
            Memory per GPU in GB, or None if TP degree not supported
        """
        if not self.validate_tp_degree(tp_degree):
            return None
        return self.memory_per_gpu.get(tp_degree, self.memory_per_gpu.get(self.recommended_tp))

    def validate_tp_degree(self, tp_degree: int) -> bool:
        """
        Validate if TP degree is supported for this model.

        Args:
            tp_degree: Tensor parallelism degree to validate

        Returns:
            True if TP degree is within valid range
        """
        return self.min_tp <= tp_degree <= self.max_tp

    def get_total_memory(self, tp_degree: int) -> Optional[float]:
        """
        Get total memory requirement for a given TP degree.

        Args:
            tp_degree: Tensor parallelism degree

        Returns:
            Total memory in GB across all GPUs, or None if invalid TP degree
        """
        mem_per_gpu = self.get_memory_for_tp(tp_degree)
        if mem_per_gpu is None:
            return None
        return mem_per_gpu * tp_degree

    def get_min_gpu_memory_requirement(self) -> float:
        """
        Get minimum GPU memory requirement (at max TP).

        Returns:
            Minimum memory per GPU in GB
        """
        return min(self.memory_per_gpu.values())


# LLM model specifications
# Memory calculations assume 8-bit quantization for model weights
# TP memory distribution: Model weights are sharded across TP GPUs, KV cache is replicated
LLM_MODEL_SPECS: Dict[str, LLMModelSpec] = {
    "DeepSeek-R1": LLMModelSpec(
        name="DeepSeek-R1",
        parameters=671.0,  # Billion parameters
        memory_per_gpu={
            1: 671.0,  # Not feasible on single GPU
            2: 336.0,  # 336 GB per GPU (2x TP)
            4: 168.0,  # 168 GB per GPU (4x TP)
            8: 84.0,   # 84 GB per GPU (8x TP) - fits on H100 96GB
        },
        min_tp=2,
        max_tp=8,
        recommended_tp=8,
        flops_per_token_prefill=2.67e12,  # ~2.67 TFLOPs per token (671B params × 2 ops × 2)
        flops_per_token_decode=1.34e11,   # ~134 GFLOPs per token (671B / 5)
        kv_cache_per_token=0.05,  # ~50 MB per token (for 8x TP)
        description="671B parameter reasoning model by DeepSeek",
    ),
    "Qwen3": LLMModelSpec(
        name="Qwen3",
        parameters=30.0,  # Billion parameters
        memory_per_gpu={
            1: 30.0,   # Single GPU feasible
            2: 15.0,   # 15 GB per GPU (2x TP)
            4: 8.0,    # 8 GB per GPU (4x TP)
        },
        min_tp=1,
        max_tp=4,
        recommended_tp=2,
        flops_per_token_prefill=1.2e11,   # ~120 GFLOPs per token
        flops_per_token_decode=6.0e9,     # ~6 GFLOPs per token
        kv_cache_per_token=0.01,  # ~10 MB per token
        description="30B parameter model by Alibaba",
    ),
    "GLM4.7": LLMModelSpec(
        name="GLM4.7",
        parameters=355.0,  # Billion parameters
        memory_per_gpu={
            1: 355.0,  # Not feasible on single GPU
            2: 178.0,  # 178 GB per GPU
            4: 89.0,   # 89 GB per GPU - fits on H100/A100 80GB with 2x
            8: 45.0,   # 45 GB per GPU
        },
        min_tp=2,
        max_tp=8,
        recommended_tp=4,
        flops_per_token_prefill=1.42e12,  # ~1.42 TFLOPs per token
        flops_per_token_decode=7.1e10,    # ~71 GFLOPs per token
        kv_cache_per_token=0.03,  # ~30 MB per token
        description="355B parameter model by Tsinghua University",
    ),
    "Llama3-70B": LLMModelSpec(
        name="Llama3-70B",
        parameters=70.0,  # Billion parameters
        memory_per_gpu={
            1: 140.0,  # With KV cache overhead
            2: 70.0,   # 70 GB per GPU - fits on A100/H100 80GB
            4: 35.0,   # 35 GB per GPU - fits on most GPUs
            8: 18.0,   # 18 GB per GPU
        },
        min_tp=1,
        max_tp=8,
        recommended_tp=4,
        flops_per_token_prefill=2.8e11,   # ~280 GFLOPs per token
        flops_per_token_decode=1.4e10,    # ~14 GFLOPs per token
        kv_cache_per_token=0.015,  # ~15 MB per token
        description="70B parameter model by Meta",
    ),
    "Llama3-8B": LLMModelSpec(
        name="Llama3-8B",
        parameters=8.0,  # Billion parameters
        memory_per_gpu={
            1: 16.0,   # Single GPU
            2: 8.0,    # 8 GB per GPU
        },
        min_tp=1,
        max_tp=2,
        recommended_tp=1,
        flops_per_token_prefill=3.2e10,   # ~32 GFLOPs per token
        flops_per_token_decode=1.6e9,     # ~1.6 GFLOPs per token
        kv_cache_per_token=0.005,  # ~5 MB per token
        description="8B parameter model by Meta",
    ),
    "Mistral-7B": LLMModelSpec(
        name="Mistral-7B",
        parameters=7.0,  # Billion parameters
        memory_per_gpu={
            1: 14.0,   # Single GPU
            2: 7.0,    # 7 GB per GPU
        },
        min_tp=1,
        max_tp=2,
        recommended_tp=1,
        flops_per_token_prefill=2.8e10,   # ~28 GFLOPs per token
        flops_per_token_decode=1.4e9,     # ~1.4 GFLOPs per token
        kv_cache_per_token=0.004,  # ~4 MB per token
        description="7B parameter model by Mistral AI",
    ),
    "Mixtral-8x7B": LLMModelSpec(
        name="Mixtral-8x7B",
        parameters=47.0,  # Billion parameters (8 experts x 7B, ~2 active)
        memory_per_gpu={
            1: 94.0,   # With routing overhead
            2: 47.0,   # 47 GB per GPU
            4: 24.0,   # 24 GB per GPU
        },
        min_tp=1,
        max_tp=4,
        recommended_tp=2,
        flops_per_token_prefill=1.9e11,   # ~190 GFLOPs per token (MoE)
        flops_per_token_decode=9.5e9,     # ~9.5 GFLOPs per token
        kv_cache_per_token=0.012,  # ~12 MB per token
        description="47B parameter MoE model by Mistral AI (8x7B)",
    ),
}


def get_model_spec(model_name: str) -> Optional[LLMModelSpec]:
    """
    Get model specification by name.

    Args:
        model_name: Name of the LLM model

    Returns:
        LLMModelSpec object if found, None otherwise
    """
    return LLM_MODEL_SPECS.get(model_name)


def get_all_model_names() -> List[str]:
    """Get list of all available model names."""
    return list(LLM_MODEL_SPECS.keys())


def get_models_fitting_gpu(gpu_memory_gb: float) -> List[str]:
    """
    Get models that can fit on a GPU with given memory capacity.

    Args:
        gpu_memory_gb: GPU memory capacity in GB

    Returns:
        List of model names that can fit on the GPU
    """
    fitting_models = []
    for model_name, spec in LLM_MODEL_SPECS.items():
        if spec.get_min_gpu_memory_requirement() <= gpu_memory_gb:
            fitting_models.append(model_name)
    return fitting_models


def get_recommended_tp_for_gpu(model_name: str, gpu_memory_gb: float) -> Optional[int]:
    """
    Get recommended TP degree for a model on a specific GPU.

    Args:
        model_name: Name of the LLM model
        gpu_memory_gb: GPU memory capacity in GB

    Returns:
        Recommended TP degree, or None if model doesn't fit
    """
    spec = get_model_spec(model_name)
    if spec is None:
        return None

    # Try recommended TP first
    mem_needed = spec.get_memory_for_tp(spec.recommended_tp)
    if mem_needed is not None and mem_needed <= gpu_memory_gb:
        return spec.recommended_tp

    # Try other TP degrees from min to max
    for tp_degree in range(spec.max_tp, spec.min_tp - 1, -1):
        mem_needed = spec.get_memory_for_tp(tp_degree)
        if mem_needed is not None and mem_needed <= gpu_memory_gb:
            return tp_degree

    return None


if __name__ == "__main__":
    # Print all model specifications
    print("LLM Model Specifications:")
    print("-" * 100)
    for name, spec in LLM_MODEL_SPECS.items():
        print(f"\n{name:20} ({spec.description})")
        print(f"  Parameters: {spec.parameters}B")
        print(f"  TP Range: {spec.min_tp}-{spec.max_tp}, Recommended: {spec.recommended_tp}")
        print(f"  Memory per GPU: {spec.memory_per_gpu}")
        print(f"  FLOPs: Prefill {spec.flops_per_token_prefill:.2e}, Decode {spec.flops_per_token_decode:.2e}")
    print("-" * 100)

    # Test GPU fitting
    print("\nModels fitting on H100 (80GB):")
    for model in get_models_fitting_gpu(80.0):
        tp = get_recommended_tp_for_gpu(model, 80.0)
        spec = get_model_spec(model)
        mem = spec.get_memory_for_tp(tp) if tp else "N/A"
        print(f"  {model:20} - TP{tp} - {mem}GB per GPU")
