# Add project root to path FIRST (before any other imports)
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
Simplified LLM GPU Scheduling Simulator (Discrete-Time).

Key difference from token-based simulator:
- Time advances in discrete steps (dt)
- Tasks complete FIRST, freeing GPUs
- THEN scheduler chooses from waiting queue (this is where prioritization matters!)
- This design ensures algorithm differences are visible

Architecture:
1. At each time step:
   a) Complete finished tasks → free GPUs
   b) Add newly arrived tasks to queue
   c) While GPUs available: scheduler.select_next(queue) → schedule
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import copy

from extend_llm_gpu_schedule.models.llm_task import LLMTask, Phase
from extend_llm_gpu_schedule.models.llm_cluster import LLMCluster
from extend_llm_gpu_schedule.algorithms.base_llm_scheduler import BaseLLMScheduler

if TYPE_CHECKING:
    from extend_llm_gpu_schedule.models.gpu import GPU


class SimConfig:
    """Simulation configuration."""
    def __init__(
        self,
        dt: float = 0.01,  # Time step in seconds
        max_time: float = 10.0,  # Maximum simulation time
    ):
        self.dt = dt
        self.max_time = max_time


@dataclass
class RunningTask:
    """Task currently running on GPUs."""
    task: LLMTask
    gpu_group: List["GPU"]
    finish_time: float
    phase: Phase


@dataclass
class SimResult:
    """Simulation result."""
    tasks: List[LLMTask]
    makespan: float
    weighted_waiting_time: float
    average_waiting_time: float
    completed_count: int
    total_count: int
    completion_rate: float
    gpu_utilization: float
    timeline: List[Dict] = field(default_factory=list)  # For debugging/visualization


class SimplifiedLLMSimulator:
    """
    Simplified discrete-time LLM GPU scheduling simulator.

    Design principles:
    1. Discrete time steps (dt) for predictable execution
    2. GPU freeing happens BEFORE scheduling (queue can have multiple tasks)
    3. Scheduler chooses from queue based on priority (this is where algorithms differ!)
    4. Tasks have simple duration model (no complex token-level batching)
    """

    def __init__(
        self,
        cluster: LLMCluster,
        scheduler: BaseLLMScheduler,
        config: SimConfig,
    ):
        self.cluster = cluster
        self.scheduler = scheduler
        self.config = config

        # Simulation state
        self.current_time = 0.0
        self.queue: List[LLMTask] = []  # Waiting tasks (not yet scheduled)
        self.running: List[RunningTask] = []  # Currently running tasks

        # Task tracking
        self.completed_tasks: List[LLMTask] = []
        self.all_tasks: List[LLMTask] = []

        # Timeline for debugging
        self.timeline: List[Dict] = []

    def run(self, tasks: List[LLMTask]) -> SimResult:
        """
        Run simulation.

        Args:
            tasks: List of tasks to schedule (should be deep copies for each algorithm)

        Returns:
            SimResult with metrics
        """
        # Reset state
        self._reset()
        self.all_tasks = copy.deepcopy(tasks)

        # Sort tasks by arrival time for efficient processing
        tasks_by_arrival = sorted(self.all_tasks, key=lambda t: t.arrival_time)
        arrival_idx = 0
        n_tasks = len(tasks_by_arrival)

        print(f"\n[SimplifiedSim] Starting simulation: {n_tasks} tasks, dt={self.config.dt}, max_time={self.config.max_time}")
        print(f"[SimplifiedSim] Cluster: {len(self.cluster.gpus)} GPUs, Scheduler: {self.scheduler.__class__.__name__}")

        # Main simulation loop
        while self.current_time < self.config.max_time:
            # Step 1: Complete finished tasks and free GPUs
            self._complete_finished_tasks()

            # Step 2: Add newly arrived tasks to queue
            while arrival_idx < n_tasks and tasks_by_arrival[arrival_idx].arrival_time <= self.current_time:
                new_task = tasks_by_arrival[arrival_idx]
                self.queue.append(new_task)
                print(f"[SimplifiedSim] t={self.current_time:.3f}: {new_task.task_id} arrived (weight={new_task.weight}, tokens={new_task.tokens})")
                arrival_idx += 1

            # Step 3: Scheduler chooses from queue (THIS IS WHERE ALGORITHMS DIFFER!)
            self._schedule_from_queue()

            # Record timeline state
            if len(self.running) > 0 or len(self.queue) > 0:
                self._record_timeline()

            # Advance time
            self.current_time += self.config.dt

        # Complete remaining tasks at end
        self._complete_finished_tasks()

        # Force-complete any remaining running tasks
        for rt in self.running:
            rt.task.completion_time = rt.finish_time
            rt.task.assigned_gpu_group = rt.gpu_group
            self.completed_tasks.append(rt.task)

        # Calculate metrics
        result = self._calculate_result()

        print(f"\n[SimplifiedSim] Simulation complete:")
        print(f"  Completed: {result.completed_count}/{result.total_count} ({result.completion_rate:.1%})")
        print(f"  Makespan: {result.makespan:.3f}s")
        print(f"  Weighted waiting time: {result.weighted_waiting_time:.3f}")
        print(f"  Avg waiting time: {result.average_waiting_time:.3f}")

        return result

    def _complete_finished_tasks(self):
        """Complete tasks that have finished and free their GPUs."""
        remaining = []
        for rt in self.running:
            if rt.finish_time <= self.current_time:
                # Task complete
                rt.task.completion_time = rt.finish_time
                rt.task.assigned_gpu_group = rt.gpu_group
                self.completed_tasks.append(rt.task)

                # Free GPUs
                for gpu in rt.gpu_group:
                    gpu.assigned_task = None
                    gpu.timeline.append((rt.finish_time, self.current_time, None))

                print(f"[SimplifiedSim] t={self.current_time:.3f}: {rt.task.task_id} completed")
            else:
                remaining.append(rt)

        self.running = remaining

    def _schedule_from_queue(self):
        """Schedule tasks from queue using scheduler's prioritization."""
        # Ensure GPU objects have assigned_task attribute
        for gpu in self.cluster.gpus:
            if not hasattr(gpu, 'assigned_task'):
                gpu.assigned_task = None

        # Count available GPUs
        available_gpus = [g for g in self.cluster.gpus if g.assigned_task is None]

        # Determine max concurrent tasks based on TP degree
        # For simplicity, assume all tasks need 2 GPUs (TP=2 for most models)
        gpus_per_task = 2
        max_concurrent = len(available_gpus) // gpus_per_task
        current_concurrent = len(self.running)

        slots_available = max_concurrent - current_concurrent

        # Schedule tasks until no more slots or queue is empty
        while slots_available > 0 and self.queue:
            # THIS IS WHERE ALGORITHMS DIFFER!
            # Scheduler prioritizes queue and returns next task
            chosen_task = self.scheduler.select_next(self.queue)

            if chosen_task is None:
                break

            # Remove from queue
            self.queue.remove(chosen_task)

            # Allocate GPUs
            needed_gpus = gpus_per_task
            allocated_gpus = available_gpus[:needed_gpus]

            # Calculate finish time based on task duration
            # Simplified model: duration scales with token count
            # Prefill (phase=1): slower, Decode (phase=0): faster
            tokens_per_second = 1000 if chosen_task.phase == Phase.PREFILL else 5000
            duration = chosen_task.tokens / tokens_per_second
            finish_time = self.current_time + duration

            # Create running task
            running_task = RunningTask(
                task=chosen_task,
                gpu_group=allocated_gpus,
                finish_time=finish_time,
                phase=chosen_task.phase,
            )
            self.running.append(running_task)

            # Mark GPUs as assigned
            for gpu in allocated_gpus:
                gpu.assigned_task = chosen_task

            # Update available GPUs and slots
            available_gpus = available_gpus[needed_gpus:]
            slots_available -= 1

            print(f"[SimplifiedSim] t={self.current_time:.3f}: {chosen_task.task_id} scheduled "
                  f"(finish at {finish_time:.3f}, queue={len(self.queue)})")

    def _record_timeline(self):
        """Record current state for debugging."""
        self.timeline.append({
            "time": self.current_time,
            "queue_size": len(self.queue),
            "running_count": len(self.running),
            "completed_count": len(self.completed_tasks),
            "available_gpus": sum(1 for g in self.cluster.gpus if g.assigned_task is None),
        })

    def _calculate_result(self) -> SimResult:
        """Calculate simulation metrics."""
        completed = self.completed_tasks
        total = self.all_tasks

        if not completed:
            return SimResult(
                tasks=completed,
                makespan=0,
                weighted_waiting_time=0,
                average_waiting_time=0,
                completed_count=0,
                total_count=len(total),
                completion_rate=0,
                gpu_utilization=0,
                timeline=self.timeline,
            )

        # Calculate waiting times
        weighted_waiting = 0
        total_waiting = 0
        makespan = 0

        for task in completed:
            waiting_time = task.completion_time - task.arrival_time
            weighted_waiting += task.weight * waiting_time
            total_waiting += waiting_time
            makespan = max(makespan, task.completion_time)

        avg_waiting = total_waiting / len(completed)
        completion_rate = len(completed) / len(total) if total else 0

        # Simple GPU utilization (time-based)
        total_gpu_time = sum(
            rt.finish_time - rt.task.start_time
            for rt in self.running
            if rt.task.start_time is not None
        )
        gpu_utilization = total_gpu_time / (makespan * len(self.cluster.gpus)) if makespan > 0 else 0

        return SimResult(
            tasks=completed,
            makespan=makespan,
            weighted_waiting_time=weighted_waiting,
            average_waiting_time=avg_waiting,
            completed_count=len(completed),
            total_count=len(total),
            completion_rate=completion_rate,
            gpu_utilization=gpu_utilization,
            timeline=self.timeline,
        )

    def _reset(self):
        """Reset simulation state."""
        self.current_time = 0.0
        self.queue.clear()
        self.running.clear()
        self.completed_tasks.clear()
        self.all_tasks.clear()
        self.timeline.clear()

        # Reset cluster - just clear GPU assignments
        for gpu in self.cluster.gpus:
            if hasattr(gpu, 'assigned_task'):
                gpu.assigned_task = None
            if hasattr(gpu, 'timeline'):
                gpu.timeline = []


def run_simplified_experiment(
    tasks: List[LLMTask],
    cluster: LLMCluster,
    schedulers: Dict[str, BaseLLMScheduler],
    config: Optional[SimConfig] = None,
) -> Dict[str, SimResult]:
    """
    Run experiment with multiple schedulers.

    Args:
        tasks: List of tasks (will be deep copied for each scheduler)
        cluster: Cluster to use (will be reset between runs)
        schedulers: Dict of {name: scheduler}
        config: Simulation config

    Returns:
        Dict of {scheduler_name: SimResult}
    """
    if config is None:
        config = SimConfig()

    results = {}

    for name, scheduler in schedulers.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        # Deep copy tasks for each scheduler
        tasks_copy = copy.deepcopy(tasks)

        # Reset cluster
        cluster_copy = copy.deepcopy(cluster)

        # Create and run simulator
        simulator = SimplifiedLLMSimulator(cluster_copy, scheduler, config)
        result = simulator.run(tasks_copy)
        results[name] = result

    return results


if __name__ == "__main__":
    # Example usage
    from extend_llm_gpu_schedule.models.llm_task import create_prefill_task
    from extend_llm_gpu_schedule.models.llm_cluster import create_cluster
    from extend_llm_gpu_schedule.algorithms.baseline.llm_fifo import FIFOScheduler
    from extend_llm_gpu_schedule.algorithms.baseline.llm_weighted_fifo import WeightedFIFOScheduler

    # Create test tasks
    tasks = [
        create_prefill_task("T1", "Qwen3", 30, 1.0, 0.0, 1000, 2),
        create_prefill_task("T2", "Qwen3", 30, 10.0, 0.0, 500, 2),  # High weight, short
        create_prefill_task("T3", "Qwen3", 30, 1.0, 0.0, 2000, 2),  # Low weight, long
    ]

    cluster = create_cluster("h100_16gpu")

    schedulers = {
        "FIFO": FIFOScheduler(cluster),
        "WeightedFIFO": WeightedFIFOScheduler(cluster),
    }

    results = run_simplified_experiment(tasks, cluster, schedulers, SimConfig(dt=0.01, max_time=5.0))

    print("\n" + "="*60)
    print("Results Comparison:")
    print("="*60)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Weighted waiting time: {result.weighted_waiting_time:.3f}")
        print(f"  Avg waiting time: {result.average_waiting_time:.3f}")
        print(f"  Completed: {result.completed_count}/{result.total_count}")
