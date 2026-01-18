"""
Throughput estimator for LLM inference tasks.

Estimates execution duration for prefill and decode phases
using the roofline model.
"""

from typing import List, Optional
from dataclasses import dataclass

from ..models.llm_task import LLMTask, Phase
from ..models.gpu_group import GPUGroup
from .roofline_model import RooflineCalculator


@dataclass
class DurationEstimate:
    """
    Result of duration estimation.

    Attributes:
        duration: Estimated duration in seconds
        throughput: Effective throughput in tokens/s
        num_tokens: Number of tokens processed
        batch_size: Effective batch size
    """
    duration: float
    throughput: float
    num_tokens: int
    batch_size: int

    def __repr__(self) -> str:
        return (f"DurationEstimate({self.duration:.4f}s, "
                f"{self.throughput:.1f} tok/s, "
                f"{self.num_tokens} tokens)")


class ThroughputEstimator:
    """
    Estimates execution duration for LLM tasks using roofline model.

    Provides methods for estimating:
    - Prefill phase duration
    - Decode phase duration
    - Batch decode duration
    """

    def __init__(self, roofline_calculator: RooflineCalculator):
        """
        Initialize throughput estimator.

        Args:
            roofline_calculator: RooflineCalculator instance
        """
        self.roofline_calculator = roofline_calculator

    def estimate_prefill_duration(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        token_count: Optional[int] = None
    ) -> DurationEstimate:
        """
        Estimate prefill phase duration.

        Prefill is typically compute-bound (high arithmetic intensity).
        Throughput scales well with multiple GPUs.

        Args:
            task: LLM task
            gpu_group: Allocated GPU group
            token_count: Number of prefill tokens (defaults to task.prefill_tokens)

        Returns:
            DurationEstimate with duration and throughput
        """
        if token_count is None:
            token_count = task.prefill_tokens

        if token_count <= 0:
            return DurationEstimate(
                duration=0.0,
                throughput=0.0,
                num_tokens=0,
                batch_size=1
            )

        gpu_model = gpu_group.gpu_model if gpu_group.gpus else "H100"
        num_gpus = len(gpu_group.gpus)

        # Get throughput from roofline model
        throughput = self.roofline_calculator.get_throughput(
            gpu_model=gpu_model,
            model_name=task.model_name,
            phase=Phase.PREFILL,
            num_gpus=num_gpus,
            batch_size=1  # No batching for prefill
        )

        if throughput is None or throughput <= 0:
            # Fallback estimation
            throughput = self._fallback_prefill_throughput(task, gpu_group)

        duration = token_count / throughput if throughput > 0 else float('inf')

        return DurationEstimate(
            duration=duration,
            throughput=throughput,
            num_tokens=token_count,
            batch_size=1
        )

    def estimate_decode_duration(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        token_count: Optional[int] = None,
        batch_size: int = 1
    ) -> DurationEstimate:
        """
        Estimate decode phase duration for a single task.

        Decode is typically memory-bound (low arithmetic intensity).
        Benefits from batching.

        Args:
            task: LLM task
            gpu_group: Allocated GPU group
            token_count: Number of decode tokens (defaults to task.decode_tokens)
            batch_size: Batch size for continuous batching

        Returns:
            DurationEstimate with duration and throughput
        """
        if token_count is None:
            token_count = task.decode_tokens

        if token_count <= 0:
            return DurationEstimate(
                duration=0.0,
                throughput=0.0,
                num_tokens=0,
                batch_size=batch_size
            )

        gpu_model = gpu_group.gpu_model if gpu_group.gpus else "H100"
        num_gpus = len(gpu_group.gpus)

        # Get throughput from roofline model
        throughput = self.roofline_calculator.get_throughput(
            gpu_model=gpu_model,
            model_name=task.model_name,
            phase=Phase.DECODE,
            num_gpus=num_gpus,
            batch_size=batch_size
        )

        if throughput is None or throughput <= 0:
            # Fallback estimation
            throughput = self._fallback_decode_throughput(task, gpu_group, batch_size)

        duration = token_count / throughput if throughput > 0 else float('inf')

        return DurationEstimate(
            duration=duration,
            throughput=throughput,
            num_tokens=token_count,
            batch_size=batch_size
        )

    def estimate_batch_decode_duration(
        self,
        tasks: List[LLMTask],
        gpu_group: GPUGroup
    ) -> DurationEstimate:
        """
        Estimate batch decode duration.

        Processes multiple decode tasks together in a batch.
        Batching improves throughput due to better memory utilization.

        Args:
            tasks: List of tasks in the batch (should all be in decode phase)
            gpu_group: GPU group processing the batch

        Returns:
            DurationEstimate for processing one decode step for all tasks
        """
        if not tasks:
            return DurationEstimate(
                duration=0.0,
                throughput=0.0,
                num_tokens=0,
                batch_size=0
            )

        # Batch size = number of tasks
        batch_size = len(tasks)

        # Process one token per task (one decode step)
        token_count = batch_size  # Each task generates 1 token

        gpu_model = gpu_group.gpu_model if gpu_group.gpus else "H100"
        num_gpus = len(gpu_group.gpus)

        # Use the first task's model (assume all tasks use same model)
        model_name = tasks[0].model_name

        # Get throughput for batched decode
        throughput = self.roofline_calculator.get_throughput(
            gpu_model=gpu_model,
            model_name=model_name,
            phase=Phase.DECODE,
            num_gpus=num_gpus,
            batch_size=batch_size
        )

        if throughput is None or throughput <= 0:
            # Fallback: estimate from first task
            base_throughput = self._fallback_decode_throughput(tasks[0], gpu_group, 1)
            # Apply batch efficiency
            batch_efficiency = min(2.0, 1.0 + 0.2 * batch_size)
            throughput = base_throughput * batch_efficiency

        duration = token_count / throughput if throughput > 0 else float('inf')

        return DurationEstimate(
            duration=duration,
            throughput=throughput,
            num_tokens=token_count,
            batch_size=batch_size
        )

    def estimate_total_task_duration(
        self,
        task: LLMTask,
        gpu_group: GPUGroup
    ) -> float:
        """
        Estimate total duration for a task (prefill + decode).

        Args:
            task: LLM task
            gpu_group: Allocated GPU group

        Returns:
            Total duration in seconds
        """
        total_duration = 0.0

        if task.prefill_tokens > 0:
            prefill_est = self.estimate_prefill_duration(task, gpu_group)
            total_duration += prefill_est.duration

        if task.decode_tokens > 0:
            # Assume no batching for single task
            decode_est = self.estimate_decode_duration(task, gpu_group, batch_size=1)
            total_duration += decode_est.duration

        return total_duration

    def _fallback_prefill_throughput(
        self,
        task: LLMTask,
        gpu_group: GPUGroup
    ) -> float:
        """
        Fallback prefill throughput estimation.

        Used when roofline model doesn't have configuration.

        Args:
            task: LLM task
            gpu_group: GPU group

        Returns:
            Estimated throughput in tokens/s
        """
        # Simple heuristic based on GPU model
        base_throughputs = {
            "H100": 1000.0,   # tokens/s per GPU
            "H100-96GB": 1000.0,
            "A100": 600.0,
            "A100-40GB": 600.0,
            "A30": 300.0,
            "L40": 500.0,
            "L4": 150.0,
        }

        gpu_model = gpu_group.gpu_model if gpu_group.gpus else "H100"
        base = base_throughputs.get(gpu_model, 300.0)

        # Scale by number of GPUs (linear for prefill)
        num_gpus = len(gpu_group.gpus)
        return base * num_gpus

    def _fallback_decode_throughput(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        batch_size: int
    ) -> float:
        """
        Fallback decode throughput estimation.

        Used when roofline model doesn't have configuration.

        Args:
            task: LLM task
            gpu_group: GPU group
            batch_size: Batch size

        Returns:
            Estimated throughput in tokens/s
        """
        # Simple heuristic based on GPU model
        base_throughputs = {
            "H100": 150.0,    # tokens/s per GPU (slower for decode)
            "H100-96GB": 150.0,
            "A100": 100.0,
            "A100-40GB": 100.0,
            "A30": 50.0,
            "L40": 80.0,
            "L4": 25.0,
        }

        gpu_model = gpu_group.gpu_model if gpu_group.gpus else "H100"
        base = base_throughputs.get(gpu_model, 50.0)

        # Scale by number of GPUs with reduced efficiency
        num_gpus = len(gpu_group.gpus)
        scaling_efficiency = 0.85 ** (num_gpus - 1)
        multi_gpu_throughput = base * num_gpus * scaling_efficiency

        # Apply batch efficiency (diminishing returns)
        batch_efficiency = min(2.5, 1.0 + 0.5 * (batch_size ** 0.5))

        return multi_gpu_throughput * batch_efficiency


def create_throughput_estimator(
    roofline_calculator: Optional[RooflineCalculator] = None
) -> ThroughputEstimator:
    """
    Create a throughput estimator.

    Args:
        roofline_calculator: Optional RooflineCalculator instance.
                           Creates default if not provided.

    Returns:
        ThroughputEstimator instance
    """
    if roofline_calculator is None:
        from .roofline_model import create_roofline_calculator
        roofline_calculator = create_roofline_calculator()

    return ThroughputEstimator(roofline_calculator)


if __name__ == "__main__":
    # Example usage
    print("Throughput Estimator Examples:")
    print("=" * 70)

    from .roofline_model import create_roofline_calculator
    from ..models.llm_task import create_prefill_task
    from ..models.gpu_group import create_gpu_group

    # Create estimator
    calculator = create_roofline_calculator()
    estimator = create_throughput_estimator(calculator)

    # Create mock GPU group
    class MockGPU:
        def __init__(self, gpu_id, model):
            self.gpu_id = gpu_id
            self.model = model

    gpu_group = create_gpu_group([MockGPU("H100-1", "H100"), MockGPU("H100-2", "H100")])

    # Example 1: Prefill estimation
    prefill_task = create_prefill_task(
        task_id="T1",
        model_name="Qwen3",
        memory=30.0,
        weight=1.0,
        arrival_time=0.0,
        tokens=2048,
        tp_degree=2
    )
    prefill_est = estimator.estimate_prefill_duration(prefill_task, gpu_group)
    print(f"\nPrefill estimation (2048 tokens on 2x H100):")
    print(f"  Duration: {prefill_est.duration:.4f}s")
    print(f"  Throughput: {prefill_est.throughput:.1f} tokens/s")

    # Example 2: Decode estimation
    decode_task = create_prefill_task(
        task_id="T2",
        model_name="Qwen3",
        memory=30.0,
        weight=1.0,
        arrival_time=0.0,
        tokens=512,
        tp_degree=2
    )
    decode_task.phase = Phase.DECODE
    decode_task.prefill_tokens = 0
    decode_task.decode_tokens = 512

    decode_est = estimator.estimate_decode_duration(decode_task, gpu_group, batch_size=4)
    print(f"\nDecode estimation (512 tokens on 2x H100, batch=4):")
    print(f"  Duration: {decode_est.duration:.4f}s")
    print(f"  Throughput: {decode_est.throughput:.1f} tokens/s")

    # Example 3: Batch decode
    batch_tasks = [decode_task] * 8  # 8 tasks in batch
    batch_est = estimator.estimate_batch_decode_duration(batch_tasks, gpu_group)
    print(f"\nBatch decode estimation (8 tasks, 2x H100):")
    print(f"  Duration per step: {batch_est.duration:.4f}s")
    print(f"  Throughput: {batch_est.throughput:.1f} tokens/s")
