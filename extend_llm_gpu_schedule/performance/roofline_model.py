"""
Roofline model performance calculator for LLM inference.

Determines whether workload is compute-bound or memory-bound
and calculates achievable throughput.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum

from ..config.roofline_config import (
    RooflinePerformance,
    get_roofline_performance,
    ROOFLINE_PERFORMANCE_DB
)
from ..config.gpu_specs import GPUSpec, get_gpu_spec
from ..config.llm_model_specs import LLMModelSpec, get_model_spec
from ..models.llm_task import Phase


class BottleneckType(Enum):
    """Type of performance bottleneck."""
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    COMMUNICATION_BOUND = "communication_bound"


@dataclass
class ThroughputResult:
    """
    Result of throughput calculation.

    Attributes:
        throughput: Tokens per second
        bottleneck: Type of bottleneck limiting performance
        efficiency: Scaling efficiency (0.0 to 1.0)
        is_compute_bound: Whether workload is compute-bound
        is_memory_bound: Whether workload is memory-bound
    """
    throughput: float  # tokens/s
    bottleneck: BottleneckType
    efficiency: float
    is_compute_bound: bool
    is_memory_bound: bool

    def __repr__(self) -> str:
        return (f"ThroughputResult({self.throughput:.1f} tok/s, "
                f"{self.bottleneck.value}, eff={self.efficiency:.2f})")


class RooflineCalculator:
    """
    Calculates performance using the roofline model.

    The roofline model determines whether a workload is:
    - Compute-bound: limited by compute peak
    - Memory-bound: limited by memory bandwidth

    For LLM inference:
    - Prefill: typically compute-bound (high arithmetic intensity)
    - Decode: typically memory-bound (low arithmetic intensity)
    """

    def __init__(
        self,
        performance_db: Optional[Dict[Tuple[str, str, str], RooflinePerformance]] = None
    ):
        """
        Initialize roofline calculator.

        Args:
            performance_db: Optional custom performance database.
                          Defaults to ROOFLINE_PERFORMANCE_DB.
        """
        self.performance_db = performance_db or ROOFLINE_PERFORMANCE_DB

    def get_throughput(
        self,
        gpu_model: str,
        model_name: str,
        phase: Phase,
        num_gpus: int = 1,
        batch_size: int = 1
    ) -> Optional[float]:
        """
        Get token throughput for given configuration.

        Args:
            gpu_model: GPU model name (e.g., "H100", "A100")
            model_name: LLM model name (e.g., "DeepSeek-R1", "Qwen3")
            phase: Phase enum (PREFILL or DECODE)
            num_gpus: Number of GPUs (for tensor parallelism)
            batch_size: Batch size (for decode batching)

        Returns:
            Throughput in tokens/second, or None if configuration not found
        """
        perf = self._get_performance(gpu_model, model_name, phase)
        if perf is None:
            return None

        # Base throughput per GPU
        base_throughput = perf.attainable_performance

        # Apply multi-GPU scaling with efficiency
        scaling_efficiency = self._calculate_scaling_efficiency(
            perf, num_gpus, phase
        )
        multi_gpu_throughput = base_throughput * num_gpus * scaling_efficiency

        # Apply batch scaling (only for decode)
        if phase == Phase.DECODE and batch_size > 1:
            batch_efficiency = self._calculate_batch_efficiency(
                perf, batch_size
            )
            multi_gpu_throughput *= batch_efficiency

        return multi_gpu_throughput

    def calculate_throughput(
        self,
        gpu_model: str,
        model_name: str,
        phase: Phase,
        num_gpus: int = 1,
        batch_size: int = 1
    ) -> Optional[ThroughputResult]:
        """
        Calculate throughput with detailed analysis.

        Args:
            gpu_model: GPU model name
            model_name: LLM model name
            phase: Phase enum
            num_gpus: Number of GPUs
            batch_size: Batch size

        Returns:
            ThroughputResult with detailed analysis, or None if config not found
        """
        perf = self._get_performance(gpu_model, model_name, phase)
        if perf is None:
            return None

        # Determine bottleneck
        is_compute_bound = perf.is_compute_bound
        is_memory_bound = perf.is_memory_bound

        if is_compute_bound:
            bottleneck = BottleneckType.COMPUTE_BOUND
        elif is_memory_bound:
            bottleneck = BottleneckType.MEMORY_BOUND
        else:
            bottleneck = BottleneckType.COMMUNICATION_BOUND

        # Calculate efficiency
        scaling_efficiency = self._calculate_scaling_efficiency(
            perf, num_gpus, phase
        )
        batch_efficiency = 1.0
        if phase == Phase.DECODE and batch_size > 1:
            batch_efficiency = self._calculate_batch_efficiency(perf, batch_size)

        total_efficiency = scaling_efficiency * batch_efficiency

        # Calculate throughput
        throughput = perf.attainable_performance * num_gpus * total_efficiency

        return ThroughputResult(
            throughput=throughput,
            bottleneck=bottleneck,
            efficiency=total_efficiency,
            is_compute_bound=is_compute_bound,
            is_memory_bound=is_memory_bound
        )

    def _get_performance(
        self,
        gpu_model: str,
        model_name: str,
        phase: Phase
    ) -> Optional[RooflinePerformance]:
        """Get roofline performance from database."""
        phase_str = phase.value if isinstance(phase, Phase) else phase
        key = (gpu_model, model_name, phase_str)
        return self.performance_db.get(key)

    def _calculate_scaling_efficiency(
        self,
        perf: RooflinePerformance,
        num_gpus: int,
        phase: Phase
    ) -> float:
        """
        Calculate multi-GPU scaling efficiency.

        Compute-bound workloads scale better than memory-bound ones.

        Args:
            perf: Roofline performance data
            num_gpus: Number of GPUs
            phase: Execution phase

        Returns:
            Scaling efficiency (0.0 to 1.0)
        """
        if num_gpus <= 1:
            return 1.0

        # Use configured efficiency if available
        if perf.multi_gpu_scaling_efficiency < 1.0:
            # Efficiency degrades with more GPUs
            return perf.multi_gpu_scaling_efficiency ** (num_gpus - 1)

        # Otherwise, calculate based on phase
        if phase == Phase.PREFILL:
            # Compute-bound: better scaling
            base_efficiency = 0.95
            penalty = 0.02 * (num_gpus - 1)
        else:
            # Memory-bound: worse scaling
            base_efficiency = 0.85
            penalty = 0.05 * (num_gpus - 1)

        return max(0.5, base_efficiency - penalty)

    def _calculate_batch_efficiency(
        self,
        perf: RooflinePerformance,
        batch_size: int
    ) -> float:
        """
        Calculate batching efficiency for decode.

        Batching has diminishing returns.

        Args:
            perf: Roofline performance data
            batch_size: Batch size

        Returns:
            Batch efficiency multiplier (1.0 or higher)
        """
        if batch_size <= 1:
            return 1.0

        # Use configured batch scaling factor
        max_scaling = perf.batch_scaling_factor

        # Square root scaling model (common for batching)
        efficiency = 1.0 + (max_scaling - 1.0) * (batch_size ** 0.5)

        # Cap at reasonable maximum
        return min(max_scaling, efficiency)

    def estimate_duration(
        self,
        gpu_model: str,
        model_name: str,
        phase: Phase,
        num_tokens: int,
        num_gpus: int = 1,
        batch_size: int = 1
    ) -> Optional[float]:
        """
        Estimate execution duration for given token count.

        Args:
            gpu_model: GPU model name
            model_name: LLM model name
            phase: Execution phase
            num_tokens: Number of tokens to process
            num_gpus: Number of GPUs
            batch_size: Batch size

        Returns:
            Duration in seconds, or None if configuration not found
        """
        throughput = self.get_throughput(gpu_model, model_name, phase, num_gpus, batch_size)
        if throughput is None or throughput <= 0:
            return None

        return num_tokens / throughput

    def get_bottleneck_type(
        self,
        gpu_model: str,
        model_name: str,
        phase: Phase
    ) -> Optional[BottleneckType]:
        """
        Determine bottleneck type for given configuration.

        Args:
            gpu_model: GPU model name
            model_name: LLM model name
            phase: Execution phase

        Returns:
            BottleneckType, or None if configuration not found
        """
        perf = self._get_performance(gpu_model, model_name, phase)
        if perf is None:
            return None

        if perf.is_compute_bound:
            return BottleneckType.COMPUTE_BOUND
        elif perf.is_memory_bound:
            return BottleneckType.MEMORY_BOUND
        else:
            return BottleneckType.COMMUNICATION_BOUND

    def get_available_configurations(self) -> list[Tuple[str, str, str]]:
        """Get list of all available (gpu_model, model_name, phase) configurations."""
        return list(self.performance_db.keys())


def create_roofline_calculator(
    custom_db: Optional[Dict[Tuple[str, str, str], RooflinePerformance]] = None
) -> RooflineCalculator:
    """
    Create a roofline calculator.

    Args:
        custom_db: Optional custom performance database

    Returns:
        RooflineCalculator instance
    """
    return RooflineCalculator(performance_db=custom_db)


if __name__ == "__main__":
    # Example usage
    print("Roofline Calculator Examples:")
    print("=" * 70)

    calculator = create_roofline_calculator()

    # Example 1: H100 + DeepSeek-R1 + Prefill
    result = calculator.calculate_throughput("H100", "DeepSeek-R1", Phase.PREFILL, num_gpus=8)
    print(f"\nH100 8x + DeepSeek-R1 + Prefill:")
    print(f"  Throughput: {result.throughput:.1f} tokens/s")
    print(f"  Bottleneck: {result.bottleneck.value}")
    print(f"  Efficiency: {result.efficiency:.2%}")

    # Example 2: H100 + DeepSeek-R1 + Decode
    result = calculator.calculate_throughput("H100", "DeepSeek-R1", Phase.DECODE, num_gpus=8, batch_size=8)
    print(f"\nH100 8x + DeepSeek-R1 + Decode (batch=8):")
    print(f"  Throughput: {result.throughput:.1f} tokens/s")
    print(f"  Bottleneck: {result.bottleneck.value}")
    print(f"  Efficiency: {result.efficiency:.2%}")

    # Example 3: Duration estimation
    duration = calculator.estimate_duration("H100", "Qwen3", Phase.PREFILL, 2048, num_gpus=2)
    print(f"\nH100 2x + Qwen3 + Prefill 2048 tokens:")
    print(f"  Duration: {duration:.4f}s")

    # Compare across GPUs
    print(f"\nThroughput Comparison (Qwen3 Decode, batch=4):")
    print("-" * 60)
    for gpu in ["H100", "A100", "A30"]:
        result = calculator.calculate_throughput(gpu, "Qwen3", Phase.DECODE, num_gpus=2, batch_size=4)
        print(f"  {gpu:8} : {result.throughput:.1f} tokens/s ({result.bottleneck.value})")
