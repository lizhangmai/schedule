"""
Base scheduler interface for LLM GPU scheduling.

Defines the unified interface that all LLM schedulers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TYPE_CHECKING

from extend_llm_gpu_schedule.models.llm_task import LLMTask, Phase
from extend_llm_gpu_schedule.models.llm_cluster import LLMCluster
from extend_llm_gpu_schedule.models.gpu_group import GPUGroup

if TYPE_CHECKING:
    from ..simulation.llm_simulator import LLMSimulator


class BaseLLMScheduler(ABC):
    """
    Base class for LLM scheduling algorithms.

    LLM scheduling differs from traditional scheduling:
    - Two-phase execution (prefill + decode)
    - Continuous batching support
    - Token-level preemption
    - Multi-GPU allocation (tensor parallelism)

    Schedulers must implement callback methods for:
    - Task arrival: Allocate resources for new tasks
    - Prefill complete: Handle phase transition
    - Decode ready: Manage batch scheduling
    - Task complete: Clean up resources
    """

    def __init__(self, cluster: LLMCluster):
        """
        Initialize LLM scheduler.

        Args:
            cluster: LLM cluster with GPU resources
        """
        self.cluster = cluster
        self.scheduled_tasks: List[LLMTask] = []

    @abstractmethod
    def on_task_arrival(
        self,
        task: LLMTask,
        current_time: float,
        simulator: "LLMSimulator"
    ) -> Optional[GPUGroup]:
        """
        Handle task arrival and allocate resources if available.

        Called when a task arrives at the system. The scheduler should
        attempt to allocate GPUs for the task's prefill phase.

        Args:
            task: LLM task that arrived
            current_time: Current simulation time
            simulator: Simulator instance (for scheduling events)

        Returns:
            GPUGroup if resources allocated immediately, None if pending
        """
        pass

    @abstractmethod
    def on_prefill_complete(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        current_time: float,
        simulator: "LLMSimulator"
    ) -> None:
        """
        Handle prefill phase completion.

        Called when a task completes its prefill phase.
        The scheduler should transition the task to decode phase.

        Args:
            task: LLM task that completed prefill
            gpu_group: GPU group that processed prefill
            current_time: Current simulation time
            simulator: Simulator instance
        """
        pass

    @abstractmethod
    def on_decode_ready(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        current_time: float,
        simulator: "LLMSimulator"
    ) -> None:
        """
        Handle task ready for decode phase.

        Called when a task enters decode phase. The scheduler should
        manage continuous batching of decode tasks.

        Args:
            task: LLM task ready for decode
            gpu_group: GPU group allocated for decode
            current_time: Current simulation time
            simulator: Simulator instance
        """
        pass

    @abstractmethod
    def select_next(self, queue: List[LLMTask]) -> Optional[LLMTask]:
        """
        Select next task from queue to schedule.

        This is the KEY method where scheduling algorithms differ.
        Used by simplified discrete-time simulator when GPUs become available.

        Args:
            queue: List of waiting tasks (not yet scheduled)

        Returns:
            The selected task, or None if no task should be scheduled
        """
        pass

    def on_task_complete(
        self,
        task: LLMTask,
        current_time: float,
        simulator: "LLMSimulator"
    ) -> None:
        """
        Handle task completion.

        Called when a task fully completes (both phases).
        Default implementation updates tracked tasks.

        Args:
            task: LLM task that completed
            current_time: Current simulation time
            simulator: Simulator instance
        """
        if task not in self.scheduled_tasks:
            self.scheduled_tasks.append(task)

    def get_algorithm_name(self) -> str:
        """
        Return algorithm name.

        Returns:
            Algorithm class name
        """
        return self.__class__.__name__

    def reset(self) -> None:
        """Reset scheduler state."""
        self.scheduled_tasks.clear()

    def get_pending_tasks(self, current_time: float) -> List[LLMTask]:
        """
        Get tasks that have arrived but not yet started.

        Args:
            current_time: Current simulation time

        Returns:
            List of pending tasks
        """
        return [
            task for task in self.scheduled_tasks
            if task.arrival_time <= current_time
            and not task.is_started()
        ]

    def get_running_prefill_tasks(self, current_time: float) -> List[LLMTask]:
        """
        Get tasks currently in prefill phase.

        Args:
            current_time: Current simulation time

        Returns:
            List of prefill tasks
        """
        return [
            task for task in self.scheduled_tasks
            if task.is_in_prefill()
        ]

    def get_running_decode_tasks(self, current_time: float) -> List[LLMTask]:
        """
        Get tasks currently in decode phase.

        Args:
            current_time: Current simulation time

        Returns:
            List of decode tasks
        """
        return [
            task for task in self.scheduled_tasks
            if task.is_in_decode()
        ]

    def allocate_gpu_group(
        self,
        task: LLMTask,
        current_time: float
    ) -> Optional[GPUGroup]:
        """
        Allocate GPU group for a task.

        Helper method for schedulers to find and allocate GPUs.

        Args:
            task: LLM task to allocate
            current_time: Current simulation time

        Returns:
            GPUGroup if allocation succeeded, None otherwise
        """
        from ..config.llm_model_specs import get_model_spec

        model_spec = get_model_spec(task.model_name)
        if model_spec is None:
            return None

        tp_degree = task.tp_degree
        memory_per_gpu = model_spec.get_memory_for_tp(tp_degree)

        if memory_per_gpu is None:
            return None

        # Find available GPU groups
        gpu_groups = self.cluster.find_gpu_groups_for_tp(
            tp_degree=tp_degree,
            memory_per_gpu=memory_per_gpu,
            gpu_model=None,  # Any GPU model
            current_time=current_time
        )

        if not gpu_groups:
            return None

        # Select best group (earliest available)
        from ..models.gpu_group import select_best_gpu_group
        return select_best_gpu_group(
            gpu_groups,
            current_time,
            memory_per_gpu,
            prefer_earliest=True
        )


class SimpleLLMScheduler(BaseLLMScheduler):
    """
    Simple scheduler for testing and development.

    Implements basic FIFO scheduling without batching.
    """

    def __init__(self, cluster: LLMCluster):
        super().__init__(cluster)
        self.pending_queue: List[LLMTask] = []

    def on_task_arrival(
        self,
        task: LLMTask,
        current_time: float,
        simulator: "LLMSimulator"
    ) -> Optional[GPUGroup]:
        """Handle task arrival with immediate allocation if possible."""
        # Add to pending queue
        self.pending_queue.append(task)
        self.scheduled_tasks.append(task)

        # Try to allocate immediately
        gpu_group = self.allocate_gpu_group(task, current_time)

        if gpu_group:
            # Schedule prefill
            simulator.schedule_prefill(task, gpu_group, current_time)
            self.pending_queue.remove(task)

        return gpu_group

    def on_prefill_complete(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        current_time: float,
        simulator: "LLMSimulator"
    ) -> None:
        """Handle prefill complete by transitioning to decode."""
        if task.decode_tokens > 0:
            # Schedule batch decode immediately
            simulator.schedule_batch_decode(gpu_group, current_time)

    def on_decode_ready(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        current_time: float,
        simulator: "LLMSimulator"
    ) -> None:
        """Handle decode ready - batch will be processed automatically."""
        pass

    def select_next(self, queue: List[LLMTask]) -> Optional[LLMTask]:
        """Select next task using FIFO (earliest arrival time)."""
        if not queue:
            return None
        # FIFO: sort by arrival time, return earliest
        queue.sort(key=lambda t: t.arrival_time)
        return queue[0]

    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.pending_queue.clear()


if __name__ == "__main__":
    # Example usage
    print("Base LLM Scheduler Example:")
    print("=" * 70)

    from ..models.llm_cluster import create_cluster_from_cluster_config
    from ..models.llm_task import create_prefill_task

    # Create cluster
    cluster = create_cluster_from_cluster_config("h100_8gpu")
    print(f"Cluster: {cluster}")

    # Create simple scheduler
    scheduler = SimpleLLMScheduler(cluster)
    print(f"Scheduler: {scheduler.get_algorithm_name()}")

    # Create test task
    task = create_prefill_task("T1", "Qwen3", 30, 1.0, 0.0, 2048, 2)
    print(f"\nTask: {task}")

    # Test allocation (without simulator)
    gpu_group = scheduler.allocate_gpu_group(task, 0.0)
    if gpu_group:
        print(f"Allocated: {gpu_group}")
    else:
        print("No GPUs available")

    print("\n" + "=" * 70)
