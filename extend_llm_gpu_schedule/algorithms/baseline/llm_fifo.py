"""
FIFO scheduler for LLM tasks.

First-In-First-Out scheduling based on arrival time.
"""

from typing import List, Optional
from ..base_llm_scheduler import BaseLLMScheduler
from ...models.llm_task import LLMTask
from ...models.llm_cluster import LLMCluster
from ...models.gpu_group import GPUGroup


class FIFOScheduler(BaseLLMScheduler):
    """
    FIFO (First-In-First-Out) scheduler for LLM tasks.

    Strategy:
    - Tasks scheduled in order of arrival time
    - Allocate GPUs on first-come-first-served basis
    - Simple, predictable, but not optimal for weighted waiting time

    This is a baseline for comparison with more sophisticated algorithms.
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
        Handle task arrival with FIFO ordering.

        Args:
            task: LLM task that arrived
            current_time: Current simulation time
            simulator: Simulator instance

        Returns:
            GPUGroup if allocated immediately, None if queued
        """
        # Add to queue (maintains FIFO order)
        self.pending_queue.append(task)
        self.scheduled_tasks.append(task)

        # Sort by arrival time
        self.pending_queue.sort(key=lambda t: t.arrival_time)

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
        Select next task from queue using FIFO.

        For simplified simulator - selects task with earliest arrival time.
        """
        if not queue:
            return None
        # FIFO: return task with earliest arrival time
        return min(queue, key=lambda t: t.arrival_time)

    def _process_queue(self, current_time: float, simulator) -> None:
        """Process pending queue and schedule as many tasks as possible."""
        remaining_tasks = []
        scheduled_any = False

        # Sort by arrival time to maintain FIFO order
        self.pending_queue.sort(key=lambda t: t.arrival_time)

        # DEBUG: Print queue state
        if self.pending_queue and any(not t.is_scheduled() for t in self.pending_queue):
            print(f"[FIFO] Processing queue at t={current_time:.2f}, {len(self.pending_queue)} tasks")

        for task in self.pending_queue:
            if task.is_scheduled():
                continue

            # Try to allocate GPUs
            gpu_group = self.allocate_gpu_group(task, current_time)

            if gpu_group:
                # Schedule prefill
                print(f"[FIFO] Scheduling {task.task_id} (arrival={task.arrival_time:.2f})")
                simulator.schedule_prefill(task, gpu_group, current_time)
                scheduled_any = True
            else:
                # Keep in queue (may be schedulable later when different GPUs free up)
                remaining_tasks.append(task)

        self.pending_queue = remaining_tasks

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.pending_queue.clear()
