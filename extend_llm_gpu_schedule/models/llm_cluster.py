"""
LLM cluster model with tensor parallelism support.

Manages GPU cluster for LLM inference with multi-GPU allocation capabilities.
"""

from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

from .gpu_group import GPUGroup, create_gpu_group, find_gpu_groups, select_best_gpu_group
from extend_llm_gpu_schedule.config.gpu_specs import get_gpu_spec, GPUSpec
from extend_llm_gpu_schedule.config.llm_model_specs import get_model_spec


@dataclass
class LLMGPU:
    """
    GPU class for LLM inference scheduling.

    Simplified version of src.models.gpu.GPU to avoid circular imports.
    Focuses on LLM-specific attributes: memory capacity and timeline tracking.

    Attributes:
        gpu_id: GPU unique identifier
        model: GPU model name (H100, A100, etc.)
        memory_capacity: Memory capacity in GB
        scaling_factor: Compute scaling factor (for compatibility)
        bandwidth: Memory bandwidth in GB/s (from GPUSpec)
        compute_peak: Peak compute in TFLOPS (from GPUSpec)
        timeline: List of (start_time, end_time, task) tuples
        assigned_task: Currently assigned task (for simplified simulator)
    """
    gpu_id: str
    model: str
    memory_capacity: float
    scaling_factor: float = 1.0
    bandwidth: float = 0.0
    compute_peak: float = 0.0
    timeline: List[Tuple[float, float, Any]] = field(default_factory=list)
    assigned_task: Optional[Any] = None


@dataclass
class LLMCluster:
    """
    GPU cluster with LLM-specific allocation capabilities.

    Supports:
    - Multi-GPU tensor parallelism allocation
    - Continuous batching (multiple tasks per GPU)
    - Phase-aware resource management
    - Heterogeneous GPU models

    Attributes:
        gpus: List of GPUs in the cluster
        gpu_specs: Mapping of GPU IDs to their specifications
        active_allocations: Active task allocations
    """
    gpus: List[LLMGPU] = field(default_factory=list)
    gpu_specs: Dict[str, GPUSpec] = field(default_factory=dict)
    active_allocations: Dict[str, GPUGroup] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize GPU index and specifications."""
        self._gpu_index = {gpu.gpu_id: gpu for gpu in self.gpus}

        # Load GPU specifications
        for gpu in self.gpus:
            spec = get_gpu_spec(gpu.model)
            if spec:
                # Add LLM-specific attributes to GPU
                gpu.bandwidth = spec.bandwidth
                gpu.compute_peak = spec.compute_peak
                self.gpu_specs[gpu.gpu_id] = spec

    @property
    def gpu_count(self) -> int:
        """Get total number of GPUs in the cluster."""
        return len(self.gpus)

    @property
    def gpu_models(self) -> Set[str]:
        """Get set of GPU models in the cluster."""
        return {gpu.model for gpu in self.gpus}

    def get_gpu(self, gpu_id: str) -> Optional[LLMGPU]:
        """
        Get GPU by ID.

        Args:
            gpu_id: GPU identifier

        Returns:
            LLMGPU object if found, None otherwise
        """
        return self._gpu_index.get(gpu_id)

    def get_gpus_by_model(self, model: str) -> List[LLMGPU]:
        """
        Get all GPUs of a specific model.

        Args:
            model: GPU model name (e.g., "H100", "A100")

        Returns:
            List of GPUs of the specified model
        """
        return [gpu for gpu in self.gpus if gpu.model == model]

    def get_total_memory(self) -> float:
        """Get total memory capacity across all GPUs in GB."""
        return sum(gpu.memory_capacity for gpu in self.gpus)

    def get_available_memory(self, time: float = 0) -> float:
        """
        Get total available memory across all GPUs at a given time.

        Args:
            time: Time to check availability

        Returns:
            Total available memory in GB
        """
        available = 0.0
        for gpu in self.gpus:
            used = 0.0
            for (start, end, task) in gpu.timeline:
                if start <= time < end:
                    used += getattr(task, "memory", 0.0) / len(getattr(task, "assigned_gpu_group", [gpu]))
            available += gpu.memory_capacity - used
        return available

    def find_gpu_groups_for_tp(
        self,
        tp_degree: int,
        memory_per_gpu: float,
        gpu_model: Optional[str] = None,
        current_time: float = 0.0
    ) -> List[GPUGroup]:
        """
        Find all possible GPU groups for tensor parallelism.

        Args:
            tp_degree: Number of GPUs needed
            memory_per_gpu: Memory required per GPU in GB
            gpu_model: Required GPU model (None for any)
            current_time: Current simulation time (for availability check)

        Returns:
            List of valid GPUGroups, sorted by availability
        """
        # Filter GPUs by memory requirement
        valid_gpus = [
            gpu for gpu in self.gpus
            if gpu.memory_capacity >= memory_per_gpu
            and (gpu_model is None or gpu.model == gpu_model)
        ]

        if len(valid_gpus) < tp_degree:
            return []

        # Group GPUs by model
        model_groups: Dict[str, List[LLMGPU]] = {}
        for gpu in valid_gpus:
            model_groups.setdefault(gpu.model, []).append(gpu)

        groups = []

        # Generate groups for each model
        for model, gpu_list in model_groups.items():
            if len(gpu_list) >= tp_degree:
                # Simple approach: take consecutive GPUs
                # More sophisticated: consider fragmentation and availability
                for i in range(len(gpu_list) - tp_degree + 1):
                    group_gpus = gpu_list[i:i + tp_degree]
                    group_id = f"{model}_group_{i}"
                    group = create_gpu_group(group_gpus, group_id)
                    groups.append(group)

        # Sort by earliest available time
        groups.sort(key=lambda g: g.get_earliest_available_time(memory_per_gpu))

        return groups

    def allocate_gpu_group(
        self,
        task,
        tp_degree: int,
        memory_per_gpu: float,
        gpu_model: Optional[str] = None,
        current_time: float = 0.0
    ) -> Optional[GPUGroup]:
        """
        Allocate a GPU group for a task.

        Args:
            task: Task to allocate
            tp_degree: Tensor parallelism degree
            memory_per_gpu: Memory required per GPU in GB
            gpu_model: Preferred GPU model (None for any)
            current_time: Current simulation time

        Returns:
            Allocated GPUGroup, or None if no resources available
        """
        groups = self.find_gpu_groups_for_tp(tp_degree, memory_per_gpu, gpu_model, current_time)

        if not groups:
            return None

        # Select best group
        best_group = select_best_gpu_group(
            groups,
            current_time,
            memory_per_gpu,
            prefer_earliest=True
        )

        if best_group:
            # Mark as allocated
            self.active_allocations[task.task_id] = best_group
            best_group.is_active = True

        return best_group

    def deallocate_gpu_group(self, task) -> None:
        """
        Deallocate GPU group for a completed task.

        Args:
            task: Task to deallocate
        """
        if task.task_id in self.active_allocations:
            group = self.active_allocations[task.task_id]
            group.is_active = False
            del self.active_allocations[task.task_id]

    def get_active_allocations(self) -> Dict[str, GPUGroup]:
        """Get all active task allocations."""
        return self.active_allocations.copy()

    def get_utilization(self, start_time: float, end_time: float) -> float:
        """
        Calculate cluster GPU utilization over a time period.

        Args:
            start_time: Start of time period
            end_time: End of time period

        Returns:
            Average utilization (0.0 to 1.0)
        """
        if not self.gpus:
            return 0.0

        total_busy = 0.0
        total_capacity = 0.0

        for gpu in self.gpus:
            gpu_busy = 0.0
            for (t_start, t_end, _) in gpu.timeline:
                overlap_start = max(t_start, start_time)
                overlap_end = min(t_end, end_time)
                if overlap_end > overlap_start:
                    gpu_busy += overlap_end - overlap_start

            total_busy += gpu_busy
            total_capacity += end_time - start_time

        if total_capacity == 0:
            return 0.0

        return total_busy / total_capacity

    def get_model_distribution(self) -> Dict[str, int]:
        """
        Get distribution of GPU models in the cluster.

        Returns:
            Dictionary mapping model name to count
        """
        distribution = {}
        for gpu in self.gpus:
            distribution[gpu.model] = distribution.get(gpu.model, 0) + 1
        return distribution

    def reset(self) -> None:
        """Reset cluster state (clear timelines and allocations)."""
        for gpu in self.gpus:
            gpu.timeline.clear()
        self.active_allocations.clear()

    def can_accommodate_model(
        self,
        model_name: str,
        tp_degree: int,
        gpu_model: Optional[str] = None
    ) -> bool:
        """
        Check if cluster can accommodate a model with given TP degree.

        Args:
            model_name: LLM model name
            tp_degree: Tensor parallelism degree
            gpu_model: Required GPU model (None for any)

        Returns:
            True if cluster can accommodate the model
        """
        model_spec = get_model_spec(model_name)
        if model_spec is None:
            return False

        if not model_spec.validate_tp_degree(tp_degree):
            return False

        memory_per_gpu = model_spec.get_memory_for_tp(tp_degree)
        if memory_per_gpu is None:
            return False

        groups = self.find_gpu_groups_for_tp(tp_degree, memory_per_gpu, gpu_model)
        return len(groups) > 0

    def __repr__(self) -> str:
        """String representation."""
        distribution = self.get_model_distribution()
        dist_str = ", ".join(f"{model}:{count}" for model, count in distribution.items())
        return f"LLMCluster(gpus={self.gpu_count}, distribution={dist_str})"


def create_llm_cluster_from_configs(
    gpu_configs: List[Dict]
) -> LLMCluster:
    """
    Create LLM cluster from GPU configuration dictionaries.

    Args:
        gpu_configs: List of GPU config dicts with keys:
                     - gpu_id: str
                     - model: str
                     - memory_capacity: float

    Returns:
        LLMCluster instance
    """
    gpus = []
    for config in gpu_configs:
        gpu = LLMGPU(
            gpu_id=config["gpu_id"],
            model=config["model"],
            memory_capacity=config["memory_capacity"],
            scaling_factor=config.get("scaling_factor", 1.0)
        )
        gpus.append(gpu)

    return LLMCluster(gpus=gpus)


def create_cluster_from_cluster_config(
    cluster_name: str
) -> Optional[LLMCluster]:
    """
    Create LLM cluster from predefined cluster configuration.

    Args:
        cluster_name: Name of cluster configuration (e.g., "h100_8gpu")

    Returns:
        LLMCluster instance, or None if configuration not found
    """
    from extend_llm_gpu_schedule.config.gpu_specs import get_gpu_spec

    # First check if it's in CLUSTER_PRESETS
    if cluster_name in CLUSTER_PRESETS:
        preset = CLUSTER_PRESETS[cluster_name]
        return create_llm_cluster_from_configs(preset["gpus"])

    # Otherwise, treat it as a pattern like "h100_8gpu"
    # Parse the pattern
    parts = cluster_name.lower().split("_")
    if len(parts) >= 2 and parts[-1].endswith("gpu"):
        gpu_model = parts[0].upper()
        gpu_count = int(parts[-1].replace("gpu", ""))

        # Get GPU spec
        gpu_spec = get_gpu_spec(gpu_model)
        if gpu_spec is None:
            return None

        # Create GPU configs
        gpu_configs = [
            {
                "gpu_id": f"{gpu_model}-{i+1}",
                "model": gpu_model,
                "memory_capacity": gpu_spec.memory_capacity
            }
            for i in range(gpu_count)
        ]
        return create_llm_cluster_from_configs(gpu_configs)

    return None


# Alias for backward compatibility
create_cluster = create_cluster_from_cluster_config


# Predefined cluster configurations
CLUSTER_PRESETS = {
    "h100_8gpu": {
        "description": "8x H100 80GB GPUs",
        "gpus": [{"gpu_id": f"H100-{i}", "model": "H100", "memory_capacity": 80} for i in range(1, 9)]
    },
    "h100_16gpu": {
        "description": "16x H100 80GB GPUs",
        "gpus": [{"gpu_id": f"H100-{i}", "model": "H100", "memory_capacity": 80} for i in range(1, 17)]
    },
    "h100_32gpu": {
        "description": "32x H100 80GB GPUs",
        "gpus": [{"gpu_id": f"H100-{i}", "model": "H100", "memory_capacity": 80} for i in range(1, 33)]
    },
    "h100_64gpu": {
        "description": "64x H100 80GB GPUs",
        "gpus": [{"gpu_id": f"H100-{i}", "model": "H100", "memory_capacity": 80} for i in range(1, 65)]
    },
    "a100_8gpu": {
        "description": "8x A100 80GB GPUs",
        "gpus": [{"gpu_id": f"A100-{i}", "model": "A100", "memory_capacity": 80} for i in range(1, 9)]
    },
    "a100_16gpu": {
        "description": "16x A100 80GB GPUs",
        "gpus": [{"gpu_id": f"A100-{i}", "model": "A100", "memory_capacity": 80} for i in range(1, 17)]
    },
    "mixed_8gpu": {
        "description": "Mixed cluster: 4x H100, 2x A100, 2x A30",
        "gpus": [
            *[{"gpu_id": f"H100-{i}", "model": "H100", "memory_capacity": 80} for i in range(1, 5)],
            *[{"gpu_id": f"A100-{i}", "model": "A100", "memory_capacity": 80} for i in range(5, 7)],
            *[{"gpu_id": f"A30-{i}", "model": "A30", "memory_capacity": 24} for i in range(7, 9)],
        ]
    },
    "mixed_32gpu": {
        "description": "Mixed large cluster: 16x H100, 12x A100, 4x A30",
        "gpus": [
            *[{"gpu_id": f"H100-{i}", "model": "H100", "memory_capacity": 80} for i in range(1, 17)],
            *[{"gpu_id": f"A100-{i}", "model": "A100", "memory_capacity": 80} for i in range(17, 29)],
            *[{"gpu_id": f"A30-{i}", "model": "A30", "memory_capacity": 24} for i in range(29, 33)],
        ]
    },
}


if __name__ == "__main__":
    # Example usage
    print("LLM Cluster Examples:")
    print("=" * 70)

    # Create cluster from preset
    for preset_name in ["h100_8gpu", "mixed_8gpu"]:
        cluster = create_llm_cluster_from_configs(CLUSTER_PRESETS[preset_name]["gpus"])
        print(f"\n{preset_name}: {cluster}")
        print(f"  Distribution: {cluster.get_model_distribution()}")
        print(f"  Total memory: {cluster.get_total_memory():.1f}GB")

        # Check model fitting
        for model_name in ["DeepSeek-R1", "Qwen3", "Llama3-8B"]:
            for tp in [1, 2, 4, 8]:
                can_fit = cluster.can_accommodate_model(model_name, tp)
                if can_fit:
                    spec = get_model_spec(model_name)
                    mem = spec.get_memory_for_gpu(tp) if spec else 0
                    print(f"    {model_name:15} TP{tp}: ✓ ({mem}GB per GPU)")
