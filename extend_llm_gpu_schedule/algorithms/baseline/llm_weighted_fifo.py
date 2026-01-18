"""
Weighted FIFO scheduler for LLM tasks.

Prioritizes tasks by weight (higher weight first), then by arrival time.
"""

from typing import List, Optional
from ..base_llm_scheduler import BaseLLMScheduler
from ...models.llm_task import LLMTask
from ...models.llm_cluster import LLMCluster
from ...models.gpu_group import GPUGroup


class WeightedFIFOScheduler(BaseLLMScheduler):
    """
    Weighted FIFO scheduler for LLM tasks.

    Strategy:
    - Prioritize by weight (higher weight first)
    - For same weight, use FIFO (arrival time)
    - Allocate GPU group with earliest available time
    - Use recommended tensor parallelism for each model

    This scheduler optimizes for weighted waiting time objective:
        minimize Σ weight × (completion_time - arrival_time)
    """

    def __init__(self, cluster: LLMCluster):
        super().__init__(cluster)
        self.pending_queue: List[LLMTask] = []

    def on_task_arrival(
        self,
        task: LLMTask,
        current_time: float,
        simulator
    ) -> Optional[GPUGroup]:
        """
        Handle task arrival with weighted FIFO ordering.

        Args:
            task: LLM task that arrived
            current_time: Current simulation time
            simulator: Simulator instance

        Returns:
            GPUGroup if allocated immediately, None if queued
        """
        # Add to queue
        self.pending_queue.append(task)
        self.scheduled_tasks.append(task)

        # Sort by weight (descending), then arrival time (ascending)
        self.pending_queue.sort(key=lambda t: (-t.weight, t.arrival_time))

        # Try to schedule from queue
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
        """Handle prefill complete - transition to decode."""
        if task.decode_tokens > 0:
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
        """Handle decode ready - batch processing managed by simulator."""
        pass

    def select_next(self, queue: List[LLMTask]) -> Optional[LLMTask]:
        """
        Select next task from queue using Weighted FIFO.

        For simplified simulator - selects task with highest weight,
        then earliest arrival time.
        """
        if not queue:
            return None
        # Weighted FIFO: highest weight first, then earliest arrival
        return max(queue, key=lambda t: (t.weight, -t.arrival_time))

    def _process_queue(self, current_time: float, simulator) -> None:
        """Process pending queue and schedule as many tasks as possible."""
        remaining_tasks = []

        # Sort by weight (descending), then arrival time (ascending)
        self.pending_queue.sort(key=lambda t: (-t.weight, t.arrival_time))

        # DEBUG: Print queue state
        if self.pending_queue and any(not t.is_scheduled() for t in self.pending_queue):
            print(f"[WeightedFIFO] Processing queue at t={current_time:.2f}, {len(self.pending_queue)} tasks")
            print(f"[WeightedFIFO] Queue order: {[f'{t.task_id}(w{t.weight})' for t in self.pending_queue if not t.is_scheduled()]}")

        for task in self.pending_queue:
            if task.is_scheduled():
                continue

            # Try to allocate GPUs
            gpu_group = self.allocate_gpu_group(task, current_time)

            if gpu_group:
                # Schedule prefill
                print(f"[WeightedFIFO] Scheduling {task.task_id} (weight={task.weight}, arrival={task.arrival_time:.2f})")
                simulator.schedule_prefill(task, gpu_group, current_time)
            else:
                # Keep in queue (maintain sorted order)
                remaining_tasks.append(task)

        # Re-sort remaining tasks to maintain priority order
        remaining_tasks.sort(key=lambda t: (-t.weight, t.arrival_time))
        self.pending_queue = remaining_tasks

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.pending_queue.clear()


class StrictWeightedFIFOScheduler(BaseLLMScheduler):
    """
    Strict Weighted FIFO scheduler - always schedules highest weight task.

    Unlike WeightedFIFOScheduler, this will wait for resources to become
    available rather than scheduling a lower-weight task.
    """

    def __init__(self, cluster: LLMCluster):
        super().__init__(cluster)
        self.pending_queue: List[LLMTask] = []

    def on_task_arrival(
        self,
        task: LLMTask,
        current_time: float,
        simulator
    ) -> Optional[GPUGroup]:
        """Handle task arrival with strict weighted priority."""
        # Add to queue
        self.pending_queue.append(task)
        self.scheduled_tasks.append(task)

        # Sort by weight (descending), then arrival time (ascending)
        self.pending_queue.sort(key=lambda t: (-t.weight, t.arrival_time))

        # Try to schedule highest priority task
        if self.pending_queue:
            next_task = self.pending_queue[0]
            if not next_task.is_scheduled():
                gpu_group = self.allocate_gpu_group(next_task, current_time)
                if gpu_group:
                    simulator.schedule_prefill(next_task, gpu_group, current_time)
                    self.pending_queue.remove(next_task)
                    return gpu_group

        return None

    def on_prefill_complete(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        current_time: float,
        simulator
    ) -> None:
        """Handle prefill complete and try to schedule next task."""
        if task.decode_tokens > 0:
            simulator.schedule_batch_decode(gpu_group, current_time)

        # Try to schedule next highest priority task
        self._schedule_next_task(current_time, simulator)

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
        Select next task using strict weighted priority.
        """
        if not queue:
            return None
        # Same as WeightedFIFO: highest weight, then earliest arrival
        return max(queue, key=lambda t: (t.weight, -t.arrival_time))

    def _schedule_next_task(self, current_time: float, simulator) -> None:
        """Schedule the next highest priority task if resources available."""
        for task in self.pending_queue:
            if not task.is_scheduled():
                gpu_group = self.allocate_gpu_group(task, current_time)
                if gpu_group:
                    simulator.schedule_prefill(task, gpu_group, current_time)
                    self.pending_queue.remove(task)
                    return

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.pending_queue.clear()
