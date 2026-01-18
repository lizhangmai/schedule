"""
SRPT (Shortest Remaining Processing Time) scheduler for LLM tasks.

Prioritizes tasks with shortest estimated remaining processing time.
"""

from typing import List, Optional, Tuple
import heapq
from ..base_llm_scheduler import BaseLLMScheduler
from ...models.llm_task import LLMTask, Phase
from ...models.llm_cluster import LLMCluster
from ...models.gpu_group import GPUGroup
from ...performance.throughput_estimator import ThroughputEstimator


class SRPTScheduler(BaseLLMScheduler):
    """
    Shortest Remaining Processing Time (SRPT) scheduler.

    Strategy:
    - Prioritize tasks with shortest estimated remaining time
    - Minimizes average completion time
    - Requires throughput estimation

    SRPT is provably optimal for minimizing average completion time
    in single-server queues. For multi-GPU with batching, it's a heuristic.
    """

    def __init__(
        self,
        cluster: LLMCluster,
        throughput_estimator: Optional[ThroughputEstimator] = None
    ):
        """
        Initialize SRPT scheduler.

        Args:
            cluster: LLM cluster
            throughput_estimator: Optional estimator for duration prediction
        """
        super().__init__(cluster)
        self.throughput_estimator = throughput_estimator

        # Priority queue: (remaining_time, arrival_time, task)
        self.pending_queue: List[Tuple[float, float, LLMTask]] = []

        # Track all tasks
        self.all_tasks: List[LLMTask] = []

    def on_task_arrival(
        self,
        task: LLMTask,
        current_time: float,
        simulator
    ) -> Optional[GPUGroup]:
        """
        Handle task arrival with SRPT ordering.

        Args:
            task: LLM task that arrived
            current_time: Current simulation time
            simulator: Simulator instance

        Returns:
            GPUGroup if allocated immediately, None if queued
        """
        # Estimate remaining time
        remaining_time = self._estimate_remaining_time(task)

        # Add to priority queue
        heapq.heappush(self.pending_queue, (remaining_time, task.arrival_time, task))
        self.all_tasks.append(task)
        self.scheduled_tasks.append(task)

        # Try to schedule shortest task first
        self._process_queue(current_time, simulator)

        # Check if this task was scheduled
        if task.is_scheduled():
            return task.assigned_gpu_group  # type: ignore
        return None

    def on_prefill_complete(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        current_time: float,
        simulator
    ) -> None:
        """Handle prefill complete."""
        if task.decode_tokens > 0:
            # Update remaining time for decode phase
            remaining_time = self._estimate_decode_time(task)
            heapq.heappush(self.pending_queue, (remaining_time, task.arrival_time, task))

            simulator.schedule_batch_decode(gpu_group, current_time)

        # Try to schedule more pending tasks
        self._process_queue(current_time, simulator)

    def on_decode_ready(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        current_time: float,
        simulator
    ) -> None:
        """Handle decode ready."""
        pass

    def select_next(self, queue: List[LLMTask]) -> Optional[LLMTask]:
        """
        Select next task from queue using SRPT.

        For simplified simulator - selects task with shortest estimated remaining time.
        """
        if not queue:
            return None
        # SRPT: shortest estimated remaining time
        return min(queue, key=lambda t: self._estimate_remaining_time(t))

    def _process_queue(self, current_time: float, simulator) -> None:
        """Process pending queue and schedule as many tasks as possible."""
        remaining_queue = []

        while self.pending_queue:
            remaining_time, arrival_time, task = heapq.heappop(self.pending_queue)

            if task.is_scheduled():
                continue

            # Try to allocate GPUs
            gpu_group = self.allocate_gpu_group(task, current_time)

            if gpu_group:
                # Schedule prefill
                simulator.schedule_prefill(task, gpu_group, current_time)
            else:
                # Keep in queue (may be schedulable later with different GPUs)
                remaining_queue.append((remaining_time, arrival_time, task))

        # Rebuild heap with remaining tasks
        self.pending_queue = remaining_queue
        heapq.heapify(self.pending_queue)

    def _estimate_remaining_time(self, task: LLMTask) -> float:
        """
        Estimate remaining processing time for task.

        Args:
            task: LLM task

        Returns:
            Estimated remaining time in seconds
        """
        from ...config.llm_model_specs import get_model_spec

        model_spec = get_model_spec(task.model_name)
        if model_spec is None:
            return float('inf')

        # Estimate based on token count and model characteristics
        total_flops = (
            task.prefill_tokens * model_spec.flops_per_token_prefill +
            task.decode_tokens * model_spec.flops_per_token_decode
        )

        # Assume baseline performance (A100-like)
        baseline_flops = 30e12  # 30 TFLOPS

        # Adjust for tensor parallelism
        tp_scaling = min(1.0, 1.0 / task.tp_degree)

        return (total_flops / baseline_flops) * tp_scaling

    def _estimate_decode_time(self, task: LLMTask) -> float:
        """Estimate decode phase duration."""
        from ...config.llm_model_specs import get_model_spec

        model_spec = get_model_spec(task.model_name)
        if model_spec is None:
            return float('inf')

        # Decode is slower (memory-bound)
        total_flops = task.decode_tokens * model_spec.flops_per_token_decode

        # Lower baseline for decode (memory bandwidth limited)
        baseline_flops = 5e12  # 5 TFLOPS effective

        tp_scaling = min(1.0, 1.0 / task.tp_degree)

        return (total_flops / baseline_flops) * tp_scaling

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.pending_queue.clear()
        self.all_tasks.clear()


class WSRPTScheduler(BaseLLMScheduler):
    """
    Weighted Shortest Remaining Processing Time scheduler.

    Strategy:
    - Prioritize by weight / remaining_time ratio
    - Optimizes for weighted completion time objective

    This is a variant of SRPT that accounts for task priorities.
    """

    def __init__(
        self,
        cluster: LLMCluster,
        throughput_estimator: Optional[ThroughputEstimator] = None
    ):
        """
        Initialize WSRPT scheduler.

        Args:
            cluster: LLM cluster
            throughput_estimator: Optional throughput estimator
        """
        super().__init__(cluster)
        self.throughput_estimator = throughput_estimator
        self.pending_queue: List[Tuple[float, float, LLMTask]] = []
        self.all_tasks: List[LLMTask] = []

    def on_task_arrival(
        self,
        task: LLMTask,
        current_time: float,
        simulator
    ) -> Optional[GPUGroup]:
        """Handle task arrival with WSRPT ordering."""
        # Calculate priority: weight / remaining_time
        remaining_time = self._estimate_remaining_time(task)
        priority = -task.weight / remaining_time if remaining_time > 0 else float('-inf')

        # Add to priority queue (max-heap via negative priority)
        heapq.heappush(self.pending_queue, (priority, task.arrival_time, task))
        self.all_tasks.append(task)
        self.scheduled_tasks.append(task)

        # Try to schedule highest priority task
        self._process_queue(current_time, simulator)

        if task.is_scheduled():
            return task.assigned_gpu_group  # type: ignore
        return None

    def on_prefill_complete(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        current_time: float,
        simulator
    ) -> None:
        """Handle prefill complete."""
        if task.decode_tokens > 0:
            remaining_time = self._estimate_decode_time(task)
            priority = -task.weight / remaining_time if remaining_time > 0 else float('-inf')
            heapq.heappush(self.pending_queue, (priority, task.arrival_time, task))

            simulator.schedule_batch_decode(gpu_group, current_time)

        self._process_queue(current_time, simulator)

    def on_decode_ready(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        current_time: float,
        simulator
    ) -> None:
        """Handle decode ready."""
        pass

    def select_next(self, queue: List[LLMTask]) -> Optional[LLMTask]:
        """
        Select next task from queue using WSRPT.

        For simplified simulator - selects task with highest weight/remaining_time ratio.
        """
        if not queue:
            return None
        # WSRPT: maximize weight / remaining_time
        def priority(t):
            rt = self._estimate_remaining_time(t)
            return t.weight / rt if rt > 0 else float('inf')
        return max(queue, key=priority)

    def _process_queue(self, current_time: float, simulator) -> None:
        """Process pending queue and schedule as many tasks as possible."""
        remaining_queue = []

        while self.pending_queue:
            priority, arrival_time, task = heapq.heappop(self.pending_queue)

            if task.is_scheduled():
                continue

            gpu_group = self.allocate_gpu_group(task, current_time)

            if gpu_group:
                simulator.schedule_prefill(task, gpu_group, current_time)
            else:
                # Keep in queue (may be schedulable later with different GPUs)
                remaining_queue.append((priority, arrival_time, task))

        # Rebuild heap with remaining tasks
        self.pending_queue = remaining_queue
        heapq.heapify(self.pending_queue)

    def _estimate_remaining_time(self, task: LLMTask) -> float:
        """Estimate remaining processing time."""
        from ...config.llm_model_specs import get_model_spec

        model_spec = get_model_spec(task.model_name)
        if model_spec is None:
            return 1.0  # Default to avoid division by zero

        total_flops = (
            task.prefill_tokens * model_spec.flops_per_token_prefill +
            task.decode_tokens * model_spec.flops_per_token_decode
        )

        baseline_flops = 30e12
        tp_scaling = min(1.0, 1.0 / task.tp_degree)

        return max(0.001, (total_flops / baseline_flops) * tp_scaling)  # Avoid zero

    def _estimate_decode_time(self, task: LLMTask) -> float:
        """Estimate decode phase duration."""
        from ...config.llm_model_specs import get_model_spec

        model_spec = get_model_spec(task.model_name)
        if model_spec is None:
            return 1.0

        total_flops = task.decode_tokens * model_spec.flops_per_token_decode
        baseline_flops = 5e12
        tp_scaling = min(1.0, 1.0 / task.tp_degree)

        return max(0.001, (total_flops / baseline_flops) * tp_scaling)

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.pending_queue.clear()
        self.all_tasks.clear()
