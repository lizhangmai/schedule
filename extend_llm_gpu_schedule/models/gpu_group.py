"""
GPU group model for multi-GPU tensor parallelism allocation.

Represents a group of GPUs allocated together for tensor parallelism.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, TYPE_CHECKING, Any
import itertools


# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from .llm_cluster import LLMGPU


@dataclass
class GPUGroup:
    """
    Represents a group of GPUs allocated for tensor parallelism.

    A single LLM task spans multiple GPUs in tensor parallelism.
    All GPUs in a group should ideally be of the same model for efficiency.

    Attributes:
        gpus: List of LLMGPU objects in the group
        group_id: Unique group identifier
        gpu_model: GPU model name (for convenience)
        is_homogeneous: Whether all GPUs are the same model
        is_active: Whether the group is currently processing a task
    """
    gpus: List[Any] = field(default_factory=list)  # LLMGPU objects
    group_id: str = ""
    is_active: bool = False

    def __post_init__(self):
        """Compute derived attributes."""
        if self.gpus:
            gpu_models = {gpu.model for gpu in self.gpus}
            object.__setattr__(self, "is_homogeneous", len(gpu_models) <= 1)
        else:
            object.__setattr__(self, "is_homogeneous", True)

    @property
    def size(self) -> int:
        """Get number of GPUs in the group."""
        return len(self.gpus)

    @property
    def gpu_model(self) -> Optional[str]:
        """Get GPU model (all GPUs should be same model)."""
        if not self.gpus:
            return None
        return self.gpus[0].model

    @property
    def gpu_ids(self) -> List[str]:
        """Get list of GPU IDs in the group."""
        return [gpu.gpu_id for gpu in self.gpus]

    def get_total_memory(self) -> float:
        """
        Get total memory capacity of the group.

        Returns:
            Sum of all GPU memory capacities in GB
        """
        return sum(gpu.memory_capacity for gpu in self.gpus)

    def get_min_memory_capacity(self) -> float:
        """
        Get minimum memory capacity among GPUs (bottleneck).

        For tensor parallelism, the limiting factor is the smallest GPU.

        Returns:
            Minimum memory capacity in GB
        """
        if not self.gpus:
            return 0.0
        return min(gpu.memory_capacity for gpu in self.gpus)

    def can_accommodate(self, memory_per_gpu: float) -> bool:
        """
        Check if all GPUs can accommodate the memory requirement.

        Args:
            memory_per_gpu: Memory required per GPU in GB

        Returns:
            True if all GPUs have enough memory
        """
        return all(gpu.memory_capacity >= memory_per_gpu for gpu in self.gpus)

    def get_available_memory_at(self, time: float) -> float:
        """
        Get minimum available memory across GPUs at a given time.

        Args:
            time: Time to check availability

        Returns:
            Minimum available memory in GB across all GPUs
        """
        if not self.gpus:
            return 0.0

        available_memories = []
        for gpu in self.gpus:
            # Calculate used memory at time
            used_memory = 0.0
            for (start, end, task) in gpu.timeline:
                if start <= time < end:
                    # Assuming task has memory attribute
                    used_memory += getattr(task, "memory", 0.0)
            available_memories.append(gpu.memory_capacity - used_memory)

        return min(available_memories) if available_memories else 0.0

    def get_bandwidth(self) -> float:
        """
        Get bottleneck bandwidth (min across GPUs).

        For memory-bound operations, the slowest GPU determines throughput.

        Returns:
            Minimum bandwidth in GB/s across all GPUs
        """
        if not self.gpus:
            return 0.0
        return min(getattr(gpu, "bandwidth", 2000) for gpu in self.gpus)

    def get_compute_peak(self) -> float:
        """
        Get total compute peak (sum across GPUs).

        For compute-bound operations, total compute capacity adds up.

        Returns:
            Total compute peak in TFLOPS across all GPUs
        """
        if not self.gpus:
            return 0.0
        return sum(getattr(gpu, "compute_peak", 100) for gpu in self.gpus)

    def get_earliest_available_time(self, memory_per_gpu: float) -> float:
        """
        Find earliest time when all GPUs in group are available.

        Args:
            memory_per_gpu: Memory required per GPU in GB

        Returns:
            Earliest time when group can accommodate the request
        """
        if not self.gpus:
            return 0.0

        max_available_time = 0.0
        for gpu in self.gpus:
            gpu_available = 0.0
            for (start, end, task) in gpu.timeline:
                # Find gap that can fit memory requirement
                if gpu_available < start:
                    # This gap can accommodate
                    if gpu.memory_capacity >= memory_per_gpu:
                        break
                gpu_available = max(gpu_available, end)

            max_available_time = max(max_available_time, gpu_available)

        return max_available_time

    def allocate_task(self, task, start_time: float, duration: float) -> None:
        """
        Allocate a task on all GPUs in the group.

        Args:
            task: Task to allocate
            start_time: When to start the task
            duration: How long the task runs
        """
        for gpu in self.gpus:
            # Add to GPU timeline
            gpu.timeline.append((start_time, start_time + duration, task))
            # Sort timeline by start time
            gpu.timeline.sort(key=lambda x: x[0])

        self.is_active = True

    def deallocate_task(self) -> None:
        """
        Mark the group as inactive (task completed).

        Note: Actual timeline entries are kept for metrics calculation.
        """
        self.is_active = False

    def get_utilization(self, start_time: float, end_time: float) -> float:
        """
        Calculate GPU utilization over a time period.

        Args:
            start_time: Start of time period
            end_time: End of time period

        Returns:
            Average utilization (0.0 to 1.0) across all GPUs
        """
        if not self.gpus:
            return 0.0

        total_busy_time = 0.0
        total_capacity = 0.0

        for gpu in self.gpus:
            gpu_busy = 0.0
            for (t_start, t_end, _) in gpu.timeline:
                # Calculate overlap with query period
                overlap_start = max(t_start, start_time)
                overlap_end = min(t_end, end_time)
                if overlap_end > overlap_start:
                    gpu_busy += overlap_end - overlap_start

            total_busy_time += gpu_busy
            total_capacity += end_time - start_time

        if total_capacity == 0:
            return 0.0

        return total_busy_time / total_capacity

    def __contains__(self, gpu_id: str) -> bool:
        """Check if GPU ID is in this group."""
        return any(gpu.gpu_id == gpu_id for gpu in self.gpus)

    def __len__(self) -> int:
        """Get group size."""
        return len(self.gpus)

    def __repr__(self) -> str:
        """String representation."""
        status = "active" if self.is_active else "idle"
        return (f"GPUGroup(id={self.group_id}, size={self.size}, "
                f"model={self.gpu_model}, status={status})")


def create_gpu_group(gpus: List[Any], group_id: str = "") -> GPUGroup:
    """
    Create a GPU group from a list of LLMGPU objects.

    Args:
        gpus: List of LLMGPU objects
        group_id: Unique group identifier

    Returns:
        GPUGroup object
    """
    if not group_id:
        gpu_ids = "_".join(gpu.gpu_id for gpu in gpus)
        group_id = f"group_{gpu_ids}"

    return GPUGroup(gpus=gpus, group_id=group_id)


def find_gpu_groups(
    available_gpus: List[Any],
    group_size: int,
    memory_per_gpu: float,
    gpu_model: Optional[str] = None,
    homogeneous_only: bool = True
) -> List[GPUGroup]:
    """
    Find all possible GPU groups of a given size.

    Args:
        available_gpus: List of available LLMGPU objects
        group_size: Number of GPUs per group
        memory_per_gpu: Memory requirement per GPU in GB
        gpu_model: Required GPU model (None for any)
        homogeneous_only: Whether to only create homogeneous groups

    Returns:
        List of valid GPUGroups
    """
    # Filter GPUs by memory requirement
    valid_gpus = [
        gpu for gpu in available_gpus
        if gpu.memory_capacity >= memory_per_gpu
        and (gpu_model is None or gpu.model == gpu_model)
    ]

    if len(valid_gpus) < group_size:
        return []

    groups = []

    if homogeneous_only:
        # Group by model
        model_gpus: dict[str, List[Any]] = {}
        for gpu in valid_gpus:
            model_gpus.setdefault(gpu.model, []).append(gpu)

        # Generate combinations for each model
        for model, gpu_list in model_gpus.items():
            if len(gpu_list) >= group_size:
                # Simple: take consecutive GPUs
                # More sophisticated: consider fragmentation
                for i in range(len(gpu_list) - group_size + 1):
                    group_gpus = gpu_list[i:i + group_size]
                    group_id = f"{model}_group_{i}"
                    groups.append(create_gpu_group(group_gpus, group_id))
    else:
        # Generate all combinations (can be expensive)
        for combo in itertools.combinations(valid_gpus, group_size):
            group_id = f"hetero_group_{len(groups)}"
            groups.append(create_gpu_group(list(combo), group_id))

    return groups


def select_best_gpu_group(
    groups: List[GPUGroup],
    current_time: float,
    memory_per_gpu: float,
    prefer_earliest: bool = True
) -> Optional[GPUGroup]:
    """
    Select the best GPU group from available options.

    Selection criteria:
    1. Must accommodate memory requirement
    2. Prefer earliest available time
    3. Prefer homogeneous groups (better scaling)

    Args:
        groups: List of available GPU groups
        current_time: Current simulation time
        memory_per_gpu: Memory required per GPU
        prefer_earliest: Whether to prefer earliest availability

    Returns:
        Best GPUGroup, or None if no valid group
    """
    valid_groups = [g for g in groups if g.can_accommodate(memory_per_gpu)]

    if not valid_groups:
        return None

    if prefer_earliest:
        # Find group with earliest available time
        best_group = min(
            valid_groups,
            key=lambda g: (g.get_earliest_available_time(memory_per_gpu), not g.is_homogeneous)
        )
        return best_group
    else:
        # Prefer homogeneous groups first
        homogeneous = [g for g in valid_groups if g.is_homogeneous]
        if homogeneous:
            return min(homogeneous, key=lambda g: g.get_earliest_available_time(memory_per_gpu))
        return min(valid_groups, key=lambda g: g.get_earliest_available_time(memory_per_gpu))


if __name__ == "__main__":
    # Example usage
    print("GPU Group Examples:")
    print("=" * 70)

    # Create mock GPU objects (simplified)
    class MockGPU:
        def __init__(self, gpu_id, model, memory_capacity):
            self.gpu_id = gpu_id
            self.model = model
            self.memory_capacity = memory_capacity
            self.timeline = []

    gpus = [
        MockGPU("H100-1", "H100", 80),
        MockGPU("H100-2", "H100", 80),
        MockGPU("H100-3", "H100", 80),
        MockGPU("H100-4", "H100", 80),
        MockGPU("A100-1", "A100", 80),
        MockGPU("A100-2", "A100", 80),
    ]

    # Find 2-GPU groups for 60GB requirement
    groups = find_gpu_groups(gpus, group_size=2, memory_per_gpu=60, gpu_model="H100")

    print(f"\nFound {len(groups)} 2-GPU groups (H100, 60GB):")
    for group in groups:
        print(f"  {group}")
        print(f"    Min memory: {group.get_min_memory_capacity()}GB")
        print(f"    GPU IDs: {group.gpu_ids}")
