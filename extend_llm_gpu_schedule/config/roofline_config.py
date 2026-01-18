"""
Roofline performance configuration for LLM inference.

Defines performance parameters for (GPU, Model, Phase) combinations.
The roofline model determines whether workload is compute-bound or memory-bound.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from extend_llm_gpu_schedule.models.llm_task import Phase


@dataclass
class RooflinePerformance:
    """
    Roofline model performance parameters for a (GPU, Model, Phase) combination.

    The roofline model defines performance bounds based on:
    - Compute peak: Maximum performance when compute-bound
    - Memory bandwidth: Maximum performance when memory-bound
    - Arithmetic intensity: Ratio of compute to memory operations (FLOPs/byte)

    Attributes:
        gpu_model: GPU model (e.g., "H100", "A100")
        model_name: LLM model name (e.g., "DeepSeek-R1", "Qwen3")
        phase: Phase (PREFILL or DECODE)
        compute_peak: Peak compute performance (TFLOPS)
        memory_bandwidth: Memory bandwidth (GB/s)
        arithmetic_intensity: FLOPs per byte (determines if compute/memory-bound)
        attainable_performance: Attainable performance (tokens/s per GPU)
        batch_scaling_factor: Throughput multiplier from batching (for decode)
        multi_gpu_scaling_efficiency: Scaling efficiency with multiple GPUs
    """
    gpu_model: str
    model_name: str
    phase: str
    compute_peak: float  # TFLOPS
    memory_bandwidth: float  # GB/s
    arithmetic_intensity: float  # FLOPs/byte
    attainable_performance: float  # tokens/s per GPU
    batch_scaling_factor: float = 1.0  # For decode batching
    multi_gpu_scaling_efficiency: float = 1.0  # Linear scaling by default

    @property
    def is_compute_bound(self) -> bool:
        """Check if workload is compute-bound based on arithmetic intensity."""
        # Calculate ridge point
        ridge = (self.compute_peak * 1e12) / (self.memory_bandwidth * 1e9)
        return self.arithmetic_intensity > ridge

    @property
    def is_memory_bound(self) -> bool:
        """Check if workload is memory-bound based on arithmetic intensity."""
        return not self.is_compute_bound

    @classmethod
    def from_config(cls, config: Dict) -> "RooflinePerformance":
        """Create from configuration dictionary."""
        return cls(
            gpu_model=config["gpu_model"],
            model_name=config["model_name"],
            phase=config["phase"],
            compute_peak=config["compute_peak"],
            memory_bandwidth=config["memory_bandwidth"],
            arithmetic_intensity=config["arithmetic_intensity"],
            attainable_performance=config["attainable_performance"],
            batch_scaling_factor=config.get("batch_scaling_factor", 1.0),
            multi_gpu_scaling_efficiency=config.get("multi_gpu_scaling_efficiency", 1.0),
        )


# Roofline performance database
# Format: (gpu_model, model_name, phase) -> RooflinePerformance
# Values are estimated based on theoretical calculations and can be calibrated
ROOFLINE_PERFORMANCE_DB: Dict[Tuple[str, str, str], RooflinePerformance] = {
    # ===== H100 Configurations =====

    # H100 + DeepSeek-R1 + Prefill
    ("H100", "DeepSeek-R1", "PREFILL"): RooflinePerformance(
        gpu_model="H100",
        model_name="DeepSeek-R1",
        phase="PREFILL",
        compute_peak=989.0,  # TFLOPS
        memory_bandwidth=3350.0,  # GB/s
        arithmetic_intensity=800.0,  # FLOPs/byte (compute-bound)
        attainable_performance=625.0,  # tokens/s per GPU (5000 with 8x TP)
        batch_scaling_factor=1.0,  # No batching benefit for prefill
        multi_gpu_scaling_efficiency=0.95,  # Good scaling for compute-bound
    ),
    # H100 + DeepSeek-R1 + Decode
    ("H100", "DeepSeek-R1", "DECODE"): RooflinePerformance(
        gpu_model="H100",
        model_name="DeepSeek-R1",
        phase="DECODE",
        compute_peak=989.0,
        memory_bandwidth=3350.0,
        arithmetic_intensity=40.0,  # FLOPs/byte (memory-bound)
        attainable_performance=100.0,  # tokens/s per GPU (800 with 8x TP + batching)
        batch_scaling_factor=2.5,  # Significant batching benefit
        multi_gpu_scaling_efficiency=0.70,  # Lower scaling for memory-bound
    ),

    # H100 + Qwen3 + Prefill
    ("H100", "Qwen3", "PREFILL"): RooflinePerformance(
        gpu_model="H100",
        model_name="Qwen3",
        phase="PREFILL",
        compute_peak=989.0,
        memory_bandwidth=3350.0,
        arithmetic_intensity=500.0,  # FLOPs/byte (compute-bound)
        attainable_performance=4000.0,  # tokens/s per GPU (smaller model)
        batch_scaling_factor=1.0,
        multi_gpu_scaling_efficiency=0.90,
    ),
    # H100 + Qwen3 + Decode
    ("H100", "Qwen3", "DECODE"): RooflinePerformance(
        gpu_model="H100",
        model_name="Qwen3",
        phase="DECODE",
        compute_peak=989.0,
        memory_bandwidth=3350.0,
        arithmetic_intensity=25.0,  # FLOPs/byte (memory-bound)
        attainable_performance=750.0,  # tokens/s per GPU
        batch_scaling_factor=3.0,  # High batching benefit for smaller models
        multi_gpu_scaling_efficiency=0.75,
    ),

    # H100 + GLM4.7 + Prefill
    ("H100", "GLM4.7", "PREFILL"): RooflinePerformance(
        gpu_model="H100",
        model_name="GLM4.7",
        phase="PREFILL",
        compute_peak=989.0,
        memory_bandwidth=3350.0,
        arithmetic_intensity=700.0,  # FLOPs/byte (compute-bound)
        attainable_performance=1200.0,  # tokens/s per GPU
        batch_scaling_factor=1.0,
        multi_gpu_scaling_efficiency=0.92,
    ),
    # H100 + GLM4.7 + Decode
    ("H100", "GLM4.7", "DECODE"): RooflinePerformance(
        gpu_model="H100",
        model_name="GLM4.7",
        phase="DECODE",
        compute_peak=989.0,
        memory_bandwidth=3350.0,
        arithmetic_intensity=35.0,  # FLOPs/byte (memory-bound)
        attainable_performance=200.0,  # tokens/s per GPU
        batch_scaling_factor=2.5,
        multi_gpu_scaling_efficiency=0.72,
    ),

    # ===== A100 Configurations =====

    # A100 + DeepSeek-R1 + Prefill
    ("A100", "DeepSeek-R1", "PREFILL"): RooflinePerformance(
        gpu_model="A100",
        model_name="DeepSeek-R1",
        phase="PREFILL",
        compute_peak=312.0,  # TFLOPS
        memory_bandwidth=2039.0,  # GB/s
        arithmetic_intensity=800.0,
        attainable_performance=312.5,  # tokens/s per GPU (2500 with 8x TP)
        batch_scaling_factor=1.0,
        multi_gpu_scaling_efficiency=0.95,
    ),
    # A100 + DeepSeek-R1 + Decode
    ("A100", "DeepSeek-R1", "DECODE"): RooflinePerformance(
        gpu_model="A100",
        model_name="DeepSeek-R1",
        phase="DECODE",
        compute_peak=312.0,
        memory_bandwidth=2039.0,
        arithmetic_intensity=40.0,
        attainable_performance=50.0,  # tokens/s per GPU (400 with 8x TP + batching)
        batch_scaling_factor=2.5,
        multi_gpu_scaling_efficiency=0.70,
    ),

    # A100 + Qwen3 + Prefill
    ("A100", "Qwen3", "PREFILL"): RooflinePerformance(
        gpu_model="A100",
        model_name="Qwen3",
        phase="PREFILL",
        compute_peak=312.0,
        memory_bandwidth=2039.0,
        arithmetic_intensity=500.0,
        attainable_performance=2000.0,  # tokens/s per GPU
        batch_scaling_factor=1.0,
        multi_gpu_scaling_efficiency=0.90,
    ),
    # A100 + Qwen3 + Decode
    ("A100", "Qwen3", "DECODE"): RooflinePerformance(
        gpu_model="A100",
        model_name="Qwen3",
        phase="DECODE",
        compute_peak=312.0,
        memory_bandwidth=2039.0,
        arithmetic_intensity=25.0,
        attainable_performance=400.0,  # tokens/s per GPU
        batch_scaling_factor=3.0,
        multi_gpu_scaling_efficiency=0.75,
    ),

    # ===== A30 Configurations =====

    # A30 + Qwen3 + Prefill
    ("A30", "Qwen3", "PREFILL"): RooflinePerformance(
        gpu_model="A30",
        model_name="Qwen3",
        phase="PREFILL",
        compute_peak=66.0,  # TFLOPS
        memory_bandwidth=933.0,  # GB/s
        arithmetic_intensity=500.0,
        attainable_performance=450.0,  # tokens/s per GPU
        batch_scaling_factor=1.0,
        multi_gpu_scaling_efficiency=0.88,
    ),
    # A30 + Qwen3 + Decode
    ("A30", "Qwen3", "DECODE"): RooflinePerformance(
        gpu_model="A30",
        model_name="Qwen3",
        phase="DECODE",
        compute_peak=66.0,
        memory_bandwidth=933.0,
        arithmetic_intensity=25.0,
        attainable_performance=100.0,  # tokens/s per GPU
        batch_scaling_factor=2.5,
        multi_gpu_scaling_efficiency=0.72,
    ),

    # ===== L40 Configurations =====

    # L40 + Qwen3 + Prefill
    ("L40", "Qwen3", "PREFILL"): RooflinePerformance(
        gpu_model="L40",
        model_name="Qwen3",
        phase="PREFILL",
        compute_peak=181.0,  # TFLOPS
        memory_bandwidth=864.0,  # GB/s
        arithmetic_intensity=500.0,
        attainable_performance=1100.0,  # tokens/s per GPU
        batch_scaling_factor=1.0,
        multi_gpu_scaling_efficiency=0.85,
    ),
    # L40 + Qwen3 + Decode
    ("L40", "Qwen3", "DECODE"): RooflinePerformance(
        gpu_model="L40",
        model_name="Qwen3",
        phase="DECODE",
        compute_peak=181.0,
        memory_bandwidth=864.0,
        arithmetic_intensity=25.0,
        attainable_performance=180.0,  # tokens/s per GPU
        batch_scaling_factor=2.5,
        multi_gpu_scaling_efficiency=0.70,
    ),
}


def get_roofline_performance(
    gpu_model: str,
    model_name: str,
    phase: Phase
) -> Optional[RooflinePerformance]:
    """
    Get roofline performance for given configuration.

    Args:
        gpu_model: GPU model name (e.g., "H100", "A100")
        model_name: LLM model name (e.g., "DeepSeek-R1", "Qwen3")
        phase: Phase enum (PREFILL or DECODE)

    Returns:
        RooflinePerformance object if found, None otherwise
    """
    phase_str = phase.value if isinstance(phase, Phase) else phase
    key = (gpu_model, model_name, phase_str)
    return ROOFLINE_PERFORMANCE_DB.get(key)


def add_roofline_performance(perf: RooflinePerformance) -> None:
    """
    Add or update roofline performance in the database.

    Args:
        perf: RooflinePerformance object to add
    """
    key = (perf.gpu_model, perf.model_name, perf.phase)
    ROOFLINE_PERFORMANCE_DB[key] = perf


def estimate_throughput(
    gpu_model: str,
    model_name: str,
    phase: Phase,
    num_gpus: int = 1,
    batch_size: int = 1
) -> Optional[float]:
    """
    Estimate token throughput for given configuration.

    Args:
        gpu_model: GPU model name
        model_name: LLM model name
        phase: Phase enum
        num_gpus: Number of GPUs (for tensor parallelism)
        batch_size: Batch size (for decode batching)

    Returns:
        Throughput in tokens/second, or None if configuration not found
    """
    perf = get_roofline_performance(gpu_model, model_name, phase)
    if perf is None:
        return None

    # Base throughput per GPU
    base_throughput = perf.attainable_performance

    # Apply multi-GPU scaling with efficiency
    scaling_efficiency = perf.multi_gpu_scaling_efficiency ** (num_gpus - 1)
    multi_gpu_throughput = base_throughput * num_gpus * scaling_efficiency

    # Apply batch scaling (only for decode)
    if phase == Phase.DECODE and batch_size > 1:
        # Batching has diminishing returns
        batch_factor = min(perf.batch_scaling_factor,
                          1.0 + (perf.batch_scaling_factor - 1.0) * (batch_size ** 0.5))
        multi_gpu_throughput *= batch_factor

    return multi_gpu_throughput


def get_all_configurations() -> List[Tuple[str, str, str]]:
    """Get list of all (gpu_model, model_name, phase) configurations."""
    return list(ROOFLINE_PERFORMANCE_DB.keys())


if __name__ == "__main__":
    # Print all configurations
    print("Roofline Performance Database:")
    print("-" * 110)
    for (gpu, model, phase), perf in ROOFLINE_PERFORMANCE_DB.items():
        bound = "Compute" if perf.is_compute_bound else "Memory"
        print(f"{gpu:8} | {model:15} | {phase:8} | {perf.attainable_performance:7.1f} tok/s | "
              f"{bound:6}-bound | AI: {perf.arithmetic_intensity:6.1f} FLOPs/byte")
    print("-" * 110)

    # Test throughput estimation
    print("\nThroughput Examples:")
    print("=" * 60)

    # H100 + DeepSeek-R1 + 8x TP
    prefill_tps = estimate_throughput("H100", "DeepSeek-R1", Phase.PREFILL, num_gpus=8)
    decode_tps = estimate_throughput("H100", "DeepSeek-R1", Phase.DECODE, num_gpus=8, batch_size=8)
    print(f"H100 8x + DeepSeek-R1:")
    print(f"  Prefill: {prefill_tps:.1f} tokens/s")
    print(f"  Decode (batch=8): {decode_tps:.1f} tokens/s")

    # A100 + Qwen3 + 2x TP
    prefill_tps = estimate_throughput("A100", "Qwen3", Phase.PREFILL, num_gpus=2)
    decode_tps = estimate_throughput("A100", "Qwen3", Phase.DECODE, num_gpus=2, batch_size=4)
    print(f"\nA100 2x + Qwen3:")
    print(f"  Prefill: {prefill_tps:.1f} tokens/s")
    print(f"  Decode (batch=4): {decode_tps:.1f} tokens/s")
