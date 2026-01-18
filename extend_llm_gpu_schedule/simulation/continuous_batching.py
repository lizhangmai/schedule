"""
Continuous batching manager for LLM inference.

Manages dynamic batching where requests can join/leave batches
at any time during the decode phase.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
from collections import defaultdict

from ..models.llm_task import LLMTask, Phase
from ..models.gpu_group import GPUGroup


@dataclass
class BatchState:
    """
    State of a decode batch on a GPU group.

    Attributes:
        gpu_group_id: ID of the GPU group
        task_ids: Set of task IDs in the batch
        max_batch_size: Maximum batch size
        current_batch_size: Current number of tasks in batch
        created_at: When this batch was created
        last_updated: When the batch was last updated
    """
    gpu_group_id: str
    task_ids: Set[str] = field(default_factory=set)
    max_batch_size: int = 32
    current_batch_size: int = 0
    created_at: float = 0.0
    last_updated: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Check if batch is empty."""
        return len(self.task_ids) == 0

    @property
    def is_full(self) -> bool:
        """Check if batch is at maximum capacity."""
        return len(self.task_ids) >= self.max_batch_size

    @property
    def utilization(self) -> float:
        """Get batch utilization (0.0 to 1.0)."""
        return self.current_batch_size / self.max_batch_size if self.max_batch_size > 0 else 0.0

    def add_task(self, task_id: str) -> bool:
        """
        Add task to batch.

        Args:
            task_id: Task to add

        Returns:
            True if added, False if batch is full
        """
        if self.is_full:
            return False
        self.task_ids.add(task_id)
        self.current_batch_size += 1
        return True

    def remove_task(self, task_id: str) -> bool:
        """
        Remove task from batch.

        Args:
            task_id: Task to remove

        Returns:
            True if removed, False if task not in batch
        """
        if task_id in self.task_ids:
            self.task_ids.remove(task_id)
            self.current_batch_size -= 1
            return True
        return False


class ContinuousBatchingManager:
    """
    Manages continuous batching for LLM inference.

    Continuous batching allows multiple requests to share GPU resources:
    - New requests can join the batch at any time
    - Completed requests leave the batch immediately
    - Batches are dynamically formed and reformed

    This is essential for efficient decode phase execution.
    """

    def __init__(self, max_batch_size: int = 32, enable_auto_cleanup: bool = True):
        """
        Initialize continuous batching manager.

        Args:
            max_batch_size: Maximum tasks per batch
            enable_auto_cleanup: Automatically clean up empty batches
        """
        self.max_batch_size = max_batch_size
        self.enable_auto_cleanup = enable_auto_cleanup

        # Active batches: gpu_group_id -> BatchState
        self.active_batches: Dict[str, BatchState] = {}

        # Task to batch mapping: task_id -> gpu_group_id
        self.task_to_batch: Dict[str, str] = {}

        # Statistics
        self.total_batches_formed = 0
        self.total_tasks_batched = 0

    def can_add_to_batch(
        self,
        task: LLMTask,
        gpu_group: GPUGroup
    ) -> bool:
        """
        Check if task can be added to existing batch on GPU group.

        Args:
            task: LLM task to add
            gpu_group: GPU group to check

        Returns:
            True if task can join the batch
        """
        if task.phase != Phase.DECODE:
            return False

        batch_id = gpu_group.group_id

        # No active batch - can always start new one
        if batch_id not in self.active_batches:
            return True

        batch = self.active_batches[batch_id]

        # Check if batch has room
        return not batch.is_full

    def add_to_batch(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        current_time: float
    ) -> bool:
        """
        Add task to batch on GPU group.

        Args:
            task: LLM task
            gpu_group: GPU group
            current_time: Current simulation time

        Returns:
            True if successfully added
        """
        if task.phase != Phase.DECODE:
            return False

        batch_id = gpu_group.group_id

        # Create new batch if needed
        if batch_id not in self.active_batches:
            self.active_batches[batch_id] = BatchState(
                gpu_group_id=batch_id,
                max_batch_size=self.max_batch_size,
                created_at=current_time
            )
            self.total_batches_formed += 1

        batch = self.active_batches[batch_id]

        # Add task to batch
        if not batch.add_task(task.task_id):
            return False

        # Update mappings
        self.task_to_batch[task.task_id] = batch_id
        batch.last_updated = current_time
        self.total_tasks_batched += 1

        return True

    def remove_from_batch(self, task: LLMTask, current_time: float) -> bool:
        """
        Remove task from its batch.

        Args:
            task: LLM task
            current_time: Current simulation time

        Returns:
            True if removed, False if task not in any batch
        """
        if task.task_id not in self.task_to_batch:
            return False

        batch_id = self.task_to_batch[task.task_id]
        batch = self.active_batches.get(batch_id)

        if batch is None:
            del self.task_to_batch[task.task_id]
            return False

        # Remove from batch
        batch.remove_task(task.task_id)
        del self.task_to_batch[task.task_id]
        batch.last_updated = current_time

        # Auto-cleanup empty batches
        if self.enable_auto_cleanup and batch.is_empty:
            del self.active_batches[batch_id]

        return True

    def get_batch_tasks(
        self,
        gpu_group: GPUGroup,
        all_tasks: Dict[str, LLMTask]
    ) -> List[LLMTask]:
        """
        Get all tasks in the batch for a GPU group.

        Args:
            gpu_group: GPU group
            all_tasks: Dictionary of all tasks

        Returns:
            List of tasks in the batch
        """
        batch_id = gpu_group.group_id

        if batch_id not in self.active_batches:
            return []

        batch = self.active_batches[batch_id]
        return [
            all_tasks[task_id]
            for task_id in batch.task_ids
            if task_id in all_tasks
        ]

    def get_batch_size(self, gpu_group: GPUGroup) -> int:
        """
        Get current batch size for GPU group.

        Args:
            gpu_group: GPU group

        Returns:
            Number of tasks in batch
        """
        batch_id = gpu_group.group_id
        batch = self.active_batches.get(batch_id)
        return batch.current_batch_size if batch else 0

    def get_batch_utilization(self, gpu_group: GPUGroup) -> float:
        """
        Get batch utilization for GPU group.

        Args:
            gpu_group: GPU group

        Returns:
            Batch utilization (0.0 to 1.0)
        """
        batch_id = gpu_group.group_id
        batch = self.active_batches.get(batch_id)
        return batch.utilization if batch else 0.0

    def should_schedule_batch(
        self,
        gpu_group: GPUGroup,
        min_batch_size: int = 1,
        max_wait_time: float = 0.01
    ) -> bool:
        """
        Determine if a batch should be scheduled.

        A batch should be scheduled when:
        - It reaches minimum size, OR
        - It has waited too long (to avoid starvation)

        Args:
            gpu_group: GPU group
            min_batch_size: Minimum tasks before scheduling
            max_wait_time: Maximum wait time before forced scheduling

        Returns:
            True if batch should be scheduled
        """
        batch_id = gpu_group.group_id
        batch = self.active_batches.get(batch_id)

        if batch is None or batch.is_empty:
            return False

        # Check if batch is large enough
        if batch.current_batch_size >= min_batch_size:
            return True

        # Check if batch has waited too long
        import time
        wait_time = time.time() - batch.last_updated
        # Note: In simulation, we'd use simulation time instead
        # For now, just check minimum size
        return False

    def get_optimal_batch_size(
        self,
        gpu_group: GPUGroup,
        current_time: float
    ) -> int:
        """
        Get optimal batch size for scheduling.

        Balances throughput (larger batches) vs latency (smaller batches).

        Args:
            gpu_group: GPU group
            current_time: Current simulation time

        Returns:
            Optimal batch size
        """
        batch_id = gpu_group.group_id
        batch = self.active_batches.get(batch_id)

        if batch is None:
            return 0

        # Simple heuristic: use all available tasks
        # More sophisticated: consider wait times, priorities
        return batch.current_batch_size

    def get_active_batch_count(self) -> int:
        """Get number of active batches."""
        return len(self.active_batches)

    def get_total_active_tasks(self) -> int:
        """Get total number of tasks across all batches."""
        return sum(batch.current_batch_size for batch in self.active_batches.values())

    def get_average_batch_utilization(self) -> float:
        """Get average batch utilization across all active batches."""
        if not self.active_batches:
            return 0.0

        total_utilization = sum(batch.utilization for batch in self.active_batches.values())
        return total_utilization / len(self.active_batches)

    def reset(self) -> None:
        """Reset all batching state."""
        self.active_batches.clear()
        self.task_to_batch.clear()
        self.total_batches_formed = 0
        self.total_tasks_batched = 0

    def get_statistics(self) -> Dict[str, any]:
        """
        Get batching statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "active_batches": len(self.active_batches),
            "total_tasks_batched": self.total_tasks_batched,
            "total_batches_formed": self.total_batches_formed,
            "average_batch_size": (
                self.get_total_active_tasks() / len(self.active_batches)
                if self.active_batches else 0
            ),
            "average_utilization": self.get_average_batch_utilization(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (f"ContinuousBatchingManager("
                f"batches={len(self.active_batches)}, "
                f"tasks={self.get_total_active_tasks()}, "
                f"avg_util={self.get_average_batch_utilization():.2%})")


def create_batching_manager(
    max_batch_size: int = 32,
    enable_auto_cleanup: bool = True
) -> ContinuousBatchingManager:
    """
    Create a continuous batching manager.

    Args:
        max_batch_size: Maximum tasks per batch
        enable_auto_cleanup: Auto-cleanup empty batches

    Returns:
        ContinuousBatchingManager instance
    """
    return ContinuousBatchingManager(max_batch_size, enable_auto_cleanup)


if __name__ == "__main__":
    # Example usage
    print("Continuous Batching Manager Examples:")
    print("=" * 70)

    from ..models.llm_task import create_decode_task

    # Create batching manager
    manager = create_batching_manager(max_batch_size=4)

    # Create mock GPU group
    class MockGPUGroup:
        def __init__(self, group_id):
            self.group_id = group_id
            self.gpus = []

    gpu_group = MockGPUGroup("group_1")

    # Create some decode tasks
    tasks = [
        create_decode_task(f"T{i}", "Qwen3", 30, 1.0, float(i) * 0.1, 512, 2)
        for i in range(1, 6)
    ]

    # Add tasks to batch
    print(f"\nAdding tasks to batch (max_size={manager.max_batch_size}):")
    for task in tasks:
        can_add = manager.can_add_to_batch(task, gpu_group)
        if can_add:
            manager.add_to_batch(task, gpu_group, current_time=task.arrival_time)
            print(f"  {task.task_id}: added (batch_size={manager.get_batch_size(gpu_group)})")
        else:
            print(f"  {task.task_id}: batch full!")

    # Get batch statistics
    print(f"\nBatch statistics: {manager.get_statistics()}")

    # Get batch tasks
    all_tasks_dict = {t.task_id: t for t in tasks}
    batch_tasks = manager.get_batch_tasks(gpu_group, all_tasks_dict)
    print(f"Tasks in batch: {[t.task_id for t in batch_tasks]}")

    # Remove some tasks
    print(f"\nRemoving T1 and T2:")
    manager.remove_from_batch(tasks[0], current_time=1.0)
    manager.remove_from_batch(tasks[1], current_time=1.0)
    print(f"  Batch size: {manager.get_batch_size(gpu_group)}")
    print(f"  Statistics: {manager.get_statistics()}")

    print("\n" + "=" * 70)
