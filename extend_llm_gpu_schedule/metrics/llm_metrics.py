"""
LLM-specific metrics calculation.

Computes performance metrics for LLM inference scheduling,
focusing on weighted waiting time as the primary objective.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict

from ..models.llm_task import LLMTask, Phase
from ..models.llm_cluster import LLMCluster


@dataclass
class LLMMetrics:
    """
    LLM-specific scheduling metrics.

    Attributes:
        weighted_waiting_time: Sum of weight * (completion - arrival)
        average_waiting_time: Average waiting time
        total_tasks: Total number of tasks
        completed_tasks: Number of completed tasks
        completion_rate: Fraction of tasks completed
        prefill_completion_time: Average prefill completion time
        decode_completion_time: Average decode completion time
        gpu_utilization: GPU time utilization (0.0 to 1.0)
        throughput: Average throughput (tasks/second)
        batch_efficiency: Average batch efficiency gain
        makespan: Total simulation time
    """
    weighted_waiting_time: float
    average_waiting_time: float
    total_tasks: int
    completed_tasks: int
    completion_rate: float
    prefill_completion_time: float
    decode_completion_time: float
    gpu_utilization: float
    throughput: float
    batch_efficiency: float
    makespan: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "weighted_waiting_time": self.weighted_waiting_time,
            "average_waiting_time": self.average_waiting_time,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "completion_rate": self.completion_rate,
            "prefill_completion_time": self.prefill_completion_time,
            "decode_completion_time": self.decode_completion_time,
            "gpu_utilization": self.gpu_utilization,
            "throughput": self.throughput,
            "batch_efficiency": self.batch_efficiency,
            "makespan": self.makespan,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (f"LLMMetrics(tasks={self.completed_tasks}/{self.total_tasks}, "
                f"weighted_wait={self.weighted_waiting_time:.2f}, "
                f"util={self.gpu_utilization:.2%})")


class LLMMetricsCalculator:
    """Calculate LLM-specific metrics."""

    @staticmethod
    def calculate(
        tasks: List[LLMTask],
        cluster: LLMCluster,
        makespan: float,
        batching_stats: Optional[Dict] = None
    ) -> LLMMetrics:
        """
        Calculate all LLM metrics.

        Args:
            tasks: List of LLM tasks (completed or not)
            cluster: LLM cluster
            makespan: Total simulation time
            batching_stats: Optional batching statistics

        Returns:
            LLMMetrics object
        """
        completed = [t for t in tasks if t.is_completed()]

        if not completed:
            return LLMMetrics(
                weighted_waiting_time=0.0,
                average_waiting_time=0.0,
                total_tasks=len(tasks),
                completed_tasks=0,
                completion_rate=0.0,
                prefill_completion_time=0.0,
                decode_completion_time=0.0,
                gpu_utilization=0.0,
                throughput=0.0,
                batch_efficiency=1.0,
                makespan=makespan,
            )

        # Primary objective: weighted waiting time
        weighted_waiting_time = sum(
            t.get_weighted_waiting_time()
            for t in completed
        )

        # Average waiting time
        average_waiting_time = sum(
            t.get_waiting_time()
            for t in completed
        ) / len(completed)

        # Phase-specific completion times
        prefill_times = [
            t.get_prefill_duration()
            for t in completed
            if t.start_time is not None and t.prefill_completion_time is not None
        ]
        decode_times = [
            t.get_decode_duration()
            for t in completed
            if t.prefill_completion_time is not None and t.completion_time is not None
        ]

        avg_prefill = sum(prefill_times) / len(prefill_times) if prefill_times else 0.0
        avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0.0

        # GPU utilization
        gpu_util = LLMMetricsCalculator._calculate_gpu_utilization(
            cluster, makespan
        )

        # Throughput
        throughput = len(completed) / makespan if makespan > 0 else 0.0

        # Batch efficiency (from stats if provided)
        batch_eff = 1.0
        if batching_stats:
            batch_eff = batching_stats.get("average_utilization", 1.0)

        return LLMMetrics(
            weighted_waiting_time=weighted_waiting_time,
            average_waiting_time=average_waiting_time,
            total_tasks=len(tasks),
            completed_tasks=len(completed),
            completion_rate=len(completed) / len(tasks),
            prefill_completion_time=avg_prefill,
            decode_completion_time=avg_decode,
            gpu_utilization=gpu_util,
            throughput=throughput,
            batch_efficiency=batch_eff,
            makespan=makespan,
        )

    @staticmethod
    def _calculate_gpu_utilization(cluster: LLMCluster, makespan: float) -> float:
        """Calculate GPU utilization."""
        if makespan == 0 or not cluster.gpus:
            return 0.0

        total_busy_time = 0.0
        for gpu in cluster.gpus:
            busy_time = sum(
                end - start
                for start, end, _ in gpu.timeline
            )
            total_busy_time += busy_time

        total_capacity = makespan * len(cluster.gpus)
        return total_busy_time / total_capacity if total_capacity > 0 else 0.0

    @staticmethod
    def calculate_per_model_metrics(
        tasks: List[LLMTask],
        cluster: LLMCluster,
        makespan: float
    ) -> Dict[str, LLMMetrics]:
        """
        Calculate metrics per LLM model.

        Args:
            tasks: List of LLM tasks
            cluster: LLM cluster
            makespan: Total simulation time

        Returns:
            Dictionary mapping model_name to LLMMetrics
        """
        # Group by model
        model_tasks: Dict[str, List[LLMTask]] = defaultdict(list)
        for task in tasks:
            model_tasks[task.model_name].append(task)

        # Calculate metrics for each model
        model_metrics = {}
        for model_name, model_task_list in model_tasks.items():
            model_metrics[model_name] = LLMMetricsCalculator.calculate(
                model_task_list, cluster, makespan
            )

        return model_metrics

    @staticmethod
    def calculate_phase_metrics(
        tasks: List[LLMTask]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate phase-specific metrics.

        Args:
            tasks: List of LLM tasks

        Returns:
            Dictionary with phase metrics
        """
        prefill_tasks = [t for t in tasks if t.prefill_tokens > 0]
        decode_tasks = [t for t in tasks if t.decode_tokens > 0]

        metrics = {
            "prefill": {
                "count": len(prefill_tasks),
                "total_tokens": sum(t.prefill_tokens for t in prefill_tasks),
                "avg_tokens": (
                    sum(t.prefill_tokens for t in prefill_tasks) / len(prefill_tasks)
                    if prefill_tasks else 0
                ),
            },
            "decode": {
                "count": len(decode_tasks),
                "total_tokens": sum(t.decode_tokens for t in decode_tasks),
                "avg_tokens": (
                    sum(t.decode_tokens for t in decode_tasks) / len(decode_tasks)
                    if decode_tasks else 0
                ),
            },
        }

        return metrics


def compare_algorithms(
    results: Dict[str, List[LLMTask]],
    cluster: LLMCluster,
    makespan: float
) -> Dict[str, Dict]:
    """
    Compare metrics across multiple algorithms.

    Args:
        results: Dictionary mapping algorithm name to completed tasks
        cluster: LLM cluster
        makespan: Simulation makespan

    Returns:
        Dictionary mapping algorithm name to metrics dict
    """
    comparison = {}

    for algo_name, tasks in results.items():
        metrics = LLMMetricsCalculator.calculate(tasks, cluster, makespan)
        comparison[algo_name] = metrics.to_dict()

    return comparison


def get_primary_objective(metrics: LLMMetrics) -> float:
    """
    Get primary objective value (weighted waiting time).

    Lower is better for optimization.

    Args:
        metrics: LLMMetrics object

    Returns:
        Weighted waiting time (primary objective)
    """
    return metrics.weighted_waiting_time


if __name__ == "__main__":
    # Example usage
    print("LLM Metrics Calculator Example:")
    print("=" * 70)

    from ..models.llm_task import create_prefill_task, create_decode_task
    from ..models.llm_cluster import create_cluster_from_cluster_config

    # Create cluster
    cluster = create_cluster_from_cluster_config("h100_8gpu")

    # Create some test tasks with simulated completion
    tasks = [
        create_prefill_task("T1", "Qwen3", 30, 1.5, 0.0, 2048, 2),
        create_prefill_task("T2", "Qwen3", 30, 1.0, 0.5, 1024, 2),
        create_decode_task("T3", "Qwen3", 30, 2.0, 1.0, 512, 2),
    ]

    # Simulate completion times
    tasks[0].start_time = 0.0
    tasks[0].prefill_completion_time = 0.5
    tasks[0].completion_time = 1.0

    tasks[1].start_time = 0.5
    tasks[1].prefill_completion_time = 0.8
    tasks[1].completion_time = 1.5

    tasks[2].start_time = 1.0
    tasks[2].prefill_completion_time = 1.0
    tasks[2].completion_time = 2.0

    # Calculate metrics
    metrics = LLMMetricsCalculator.calculate(tasks, cluster, makespan=2.0)

    print(f"\nMetrics: {metrics}")
    print(f"Primary objective (weighted waiting time): {get_primary_objective(metrics):.2f}")

    # Detailed output
    print("\nDetailed metrics:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value}")

    # Per-model metrics
    per_model = LLMMetricsCalculator.calculate_per_model_metrics(tasks, cluster, 2.0)
    print("\nPer-model metrics:")
    for model, model_metrics in per_model.items():
        print(f"  {model}: {model_metrics}")

    print("\n" + "=" * 70)
