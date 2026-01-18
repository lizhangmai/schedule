"""
LLM simulator with token-level event processing.

Implements event-driven simulation for LLM inference scheduling
with continuous batching and phase-aware execution.
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import heapq

from ..models.llm_task import LLMTask, Phase
from ..models.llm_cluster import LLMCluster
from ..models.gpu_group import GPUGroup
from .token_event import TokenEvent, TokenType, TokenEventQueue, create_task_arrival_event
from .continuous_batching import ContinuousBatchingManager
from ..performance.throughput_estimator import ThroughputEstimator
from ..performance.roofline_model import RooflineCalculator


@dataclass
class SimulatorState:
    """
    Current state of the simulator.

    Attributes:
        current_time: Current simulation time
        pending_tasks: Tasks that have arrived but not yet scheduled
        running_tasks: Tasks currently being processed
        completed_tasks: Tasks that have completed
        total_tasks: Total number of tasks to process
    """
    current_time: float = 0.0
    pending_tasks: List[LLMTask] = field(default_factory=list)
    running_tasks: Dict[str, LLMTask] = field(default_factory=dict)
    completed_tasks: List[LLMTask] = field(default_factory=list)
    total_tasks: int = 0

    @property
    def completed_count(self) -> int:
        """Get number of completed tasks."""
        return len(self.completed_tasks)

    @property
    def completion_rate(self) -> float:
        """Get completion rate (0.0 to 1.0)."""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_count / self.total_tasks


@dataclass
class SimulatorResult:
    """
    Results from LLM simulation.

    Attributes:
        tasks: All tasks with scheduling information
        completed_tasks: Only completed tasks
        makespan: Total simulation time
        weighted_waiting_time: Sum of weight * (completion - arrival)
        average_waiting_time: Average waiting time
        completion_rate: Fraction of tasks that completed
        state: Final simulator state
    """
    tasks: List[LLMTask]
    completed_tasks: List[LLMTask]
    makespan: float
    weighted_waiting_time: float
    average_waiting_time: float
    completion_rate: float
    state: SimulatorState


class LLMSimulator:
    """
    LLM-specific simulator with token-level event processing.

    Supports:
    - Token-level events for fine-grained scheduling
    - Continuous batching for decode phase
    - Phase-aware execution (prefill vs decode)
    - Multi-GPU tensor parallelism
    """

    def __init__(
        self,
        cluster: LLMCluster,
        throughput_estimator: ThroughputEstimator,
        batching_manager: Optional[ContinuousBatchingManager] = None
    ):
        """
        Initialize LLM simulator.

        Args:
            cluster: LLM cluster with GPU resources
            throughput_estimator: ThroughputEstimator for performance calculation
            batching_manager: Optional batching manager (creates default if None)
        """
        self.cluster = cluster
        self.throughput_estimator = throughput_estimator
        self.batching_manager = batching_manager or ContinuousBatchingManager()

        # Event queue
        self.event_queue = TokenEventQueue()

        # Simulator state
        self.state = SimulatorState()

        # Event handlers
        self._handlers: Dict[TokenType, Callable] = {
            TokenType.TASK_ARRIVAL: self._handle_task_arrival,
            TokenType.PREFILL_START: self._handle_prefill_start,
            TokenType.PREFILL_COMPLETE: self._handle_prefill_complete,
            TokenType.DECODE_START: self._handle_decode_start,
            TokenType.BATCH_DECODE: self._handle_batch_decode,
            TokenType.TASK_COMPLETE: self._handle_task_complete,
        }

    def run(
        self,
        tasks: List[LLMTask],
        scheduler,
        max_time: float = float('inf')
    ) -> SimulatorResult:
        """
        Run LLM scheduling simulation.

        Args:
            tasks: List of LLM tasks to schedule
            scheduler: LLM scheduler instance
            max_time: Maximum simulation time (for debugging)

        Returns:
            SimulatorResult with all scheduling information
        """
        self.reset()
        self.state.total_tasks = len(tasks)

        # Initialize: create task arrival events
        for task in tasks:
            event = create_task_arrival_event(task)
            self.event_queue.push(event)

        # Event loop
        while not self.event_queue.is_empty():
            # Get next event
            event = self.event_queue.pop()

            # Update simulation time
            self.state.current_time = event.timestamp

            # Check max time limit
            if self.state.current_time > max_time:
                break

            # Dispatch to handler
            handler = self._handlers.get(event.event_type)
            if handler:
                handler(event, scheduler)
            else:
                # No handler - ignore event
                pass

        # Compute results
        return self._compute_result()

    def _handle_task_arrival(self, event: TokenEvent, scheduler) -> None:
        """Handle task arrival event."""
        task = event.task
        if task is None:
            return

        # Add to pending tasks
        self.state.pending_tasks.append(task)

        # Invoke scheduler to make allocation decision
        # The scheduler will call back to allocate resources
        allocation = scheduler.on_task_arrival(task, self.state.current_time, self)

        if allocation is None:
            # No resources available - task remains pending
            # Scheduler may requeue it
            pass

    def _handle_prefill_start(self, event: TokenEvent, scheduler) -> None:
        """Handle prefill start event."""
        task = event.task
        gpu_group = event.gpu_group

        if task is None or gpu_group is None:
            return

        # Mark task as started
        task.start_time = event.timestamp
        task.current_phase = Phase.PREFILL
        task.assigned_gpu_group = gpu_group.gpus

        # Move from pending to running
        if task in self.state.pending_tasks:
            self.state.pending_tasks.remove(task)
        self.state.running_tasks[task.task_id] = task

        # Calculate prefill duration
        estimate = self.throughput_estimator.estimate_prefill_duration(
            task, gpu_group, task.prefill_tokens
        )

        # Schedule prefill complete event
        completion_time = event.timestamp + estimate.duration

        from .token_event import create_prefill_complete_event
        complete_event = create_prefill_complete_event(
            task, gpu_group, completion_time
        )
        self.event_queue.push(complete_event)

    def _handle_prefill_complete(self, event: TokenEvent, scheduler) -> None:
        """Handle prefill complete event."""
        task = event.task
        gpu_group = event.gpu_group

        if task is None or gpu_group is None:
            return

        # Update task state
        task.prefill_completion_time = event.timestamp

        # If no decode phase, mark task as complete
        if task.decode_tokens == 0:
            task.completion_time = event.timestamp
            task.current_phase = None
            self.state.running_tasks.pop(task.task_id, None)
            self.state.completed_tasks.append(task)

            # Deallocate GPU group
            self.cluster.deallocate_gpu_group(task)

            # Schedule task complete event
            from .token_event import create_task_complete_event
            complete_event = create_task_complete_event(
                task, gpu_group, event.timestamp
            )
            self.event_queue.push(complete_event)
        else:
            # Transition to decode phase
            task.transition_to_decode()

            # Notify scheduler
            scheduler.on_prefill_complete(task, gpu_group, event.timestamp, self)

            # Schedule decode start event
            from .token_event import create_decode_start_event
            decode_event = create_decode_start_event(
                task, gpu_group, event.timestamp
            )
            self.event_queue.push(decode_event)

    def _handle_decode_start(self, event: TokenEvent, scheduler) -> None:
        """Handle decode start event."""
        task = event.task
        gpu_group = event.gpu_group

        if task is None or gpu_group is None:
            return

        # Add to continuous batching
        self.batching_manager.add_to_batch(task, gpu_group, event.timestamp)

        # Invoke scheduler for batch scheduling
        scheduler.on_decode_ready(task, gpu_group, event.timestamp, self)

    def _handle_batch_decode(self, event: TokenEvent, scheduler) -> None:
        """Handle batch decode event."""
        gpu_group = event.gpu_group
        timestamp = event.timestamp

        if gpu_group is None:
            return

        # Get all tasks in batch
        all_tasks = {t.task_id: t for t in self.state.completed_tasks}
        all_tasks.update(self.state.running_tasks)
        all_tasks.update({t.task_id: t for t in self.state.pending_tasks})

        batch_tasks = self.batching_manager.get_batch_tasks(gpu_group, all_tasks)

        if not batch_tasks:
            return

        # Calculate batch decode duration
        estimate = self.throughput_estimator.estimate_batch_decode_duration(
            batch_tasks, gpu_group
        )

        # Process one token per task
        for task in batch_tasks:
            if task.remaining_tokens > 0:
                task.remaining_tokens -= 1

                # Check if task completed
                if task.remaining_tokens == 0:
                    task.completion_time = timestamp + estimate.duration
                    task.current_phase = None

                    # Remove from batch
                    self.batching_manager.remove_from_batch(task, timestamp + estimate.duration)

                    # Move to completed
                    self.state.running_tasks.pop(task.task_id, None)
                    self.state.completed_tasks.append(task)

                    # Deallocate GPU group
                    self.cluster.deallocate_gpu_group(task)

                    # Schedule task complete event
                    from .token_event import create_task_complete_event
                    complete_event = create_task_complete_event(
                        task, gpu_group, timestamp + estimate.duration
                    )
                    self.event_queue.push(complete_event)

        # Schedule next batch decode if there are still tasks
        remaining_tasks = self.batching_manager.get_batch_tasks(gpu_group, all_tasks)
        if remaining_tasks:
            next_event = TokenEvent(
                timestamp=timestamp + estimate.duration,
                event_type=TokenType.BATCH_DECODE,
                task=remaining_tasks[0],
                gpu_group=gpu_group,
                token_count=len(remaining_tasks),
                phase=Phase.DECODE
            )
            self.event_queue.push(next_event)

    def _handle_task_complete(self, event: TokenEvent, scheduler) -> None:
        """Handle task complete event."""
        task = event.task
        if task is None:
            return

        # Notify scheduler
        scheduler.on_task_complete(task, event.timestamp, self)

    def _compute_result(self) -> SimulatorResult:
        """Compute and return simulation results."""
        # Calculate makespan
        makespan = self.state.current_time

        # Calculate metrics
        completed = self.state.completed_tasks
        weighted_waiting_time = sum(t.get_weighted_waiting_time() for t in completed)
        average_waiting_time = (
            sum(t.get_waiting_time() for t in completed) / len(completed)
            if completed else 0.0
        )

        return SimulatorResult(
            tasks=self.state.pending_tasks + list(self.state.running_tasks.values()) + completed,
            completed_tasks=completed,
            makespan=makespan,
            weighted_waiting_time=weighted_waiting_time,
            average_waiting_time=average_waiting_time,
            completion_rate=self.state.completion_rate,
            state=self.state
        )

    def schedule_prefill(
        self,
        task: LLMTask,
        gpu_group: GPUGroup,
        start_time: float
    ) -> None:
        """
        Schedule prefill phase for a task.

        Called by scheduler to initiate prefill.

        Args:
            task: LLM task
            gpu_group: Allocated GPU group
            start_time: When to start prefill
        """
        # Allocate GPU group
        self.cluster.allocate_gpu_group(
            task, task.tp_degree,
            task.memory / task.tp_degree,
            gpu_group.gpu_model,
            start_time
        )

        # Schedule prefill start event
        from .token_event import create_prefill_start_event
        event = create_prefill_start_event(task, gpu_group, start_time)
        self.event_queue.push(event)

    def schedule_batch_decode(
        self,
        gpu_group: GPUGroup,
        timestamp: float
    ) -> None:
        """
        Schedule batch decode for a GPU group.

        Args:
            gpu_group: GPU group with decode batch
            timestamp: When to schedule
        """
        # Get batch size
        batch_size = self.batching_manager.get_batch_size(gpu_group)

        # Create batch decode event
        event = TokenEvent(
            timestamp=timestamp,
            event_type=TokenType.BATCH_DECODE,
            gpu_group=gpu_group,
            token_count=batch_size,
            phase=Phase.DECODE
        )
        self.event_queue.push(event)

    def reset(self) -> None:
        """Reset simulator state."""
        self.state = SimulatorState()
        self.event_queue.clear()
        self.batching_manager.reset()
        self.cluster.reset()

    def get_current_time(self) -> float:
        """Get current simulation time."""
        return self.state.current_time


if __name__ == "__main__":
    # Example usage
    print("LLM Simulator Example:")
    print("=" * 70)

    from ..models.llm_task import create_prefill_task, create_decode_task
    from ..models.llm_cluster import create_cluster_from_cluster_config
    from ..performance.roofline_model import create_roofline_calculator
    from ..performance.throughput_estimator import create_throughput_estimator

    # Create cluster
    cluster = create_cluster_from_cluster_config("h100_8gpu")
    print(f"Cluster: {cluster}")

    # Create performance estimator
    calculator = create_roofline_calculator()
    estimator = create_throughput_estimator(calculator)

    # Create simulator
    simulator = LLMSimulator(cluster, estimator)

    # Create some test tasks
    tasks = [
        create_prefill_task("T1", "Qwen3", 30, 1.0, 0.0, 2048, 2),
        create_prefill_task("T2", "Qwen3", 30, 1.0, 0.1, 1024, 2),
    ]

    print(f"\nTasks: {len(tasks)}")

    # Note: Need a scheduler to actually run
    # This is just to show the structure
    print("\nSimulator initialized and ready to run.")
    print("Note: Requires scheduler implementation to execute.")
    print("=" * 70)
