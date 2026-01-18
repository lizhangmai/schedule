"""
GPU specifications for LLM inference scheduling.

Based on modern NVIDIA GPU specifications for H100, A100, A30, L40.
Includes memory capacity, bandwidth, and compute peak for roofline model.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GPUSpec:
    """
    Modern GPU specification for LLM inference.

    Attributes:
        model: GPU model name (e.g., "H100", "A100")
        memory_capacity: Total memory capacity in GB
        bandwidth: Memory bandwidth in GB/s
        compute_peak: Peak compute performance in TFLOPS (FP16 with Tensor Cores)
        hbm_speed: HBM/GDDR speed type
        interconnect: GPU-to-GPU interconnect type
        nvlink_bandwidth: NVLink/interconnect bandwidth in GB/s
        tdp: Thermal design power in Watts
    """
    model: str
    memory_capacity: float  # GB
    bandwidth: float  # GB/s
    compute_peak: float  # TFLOPS (FP16 with Tensor Cores)
    hbm_speed: str
    interconnect: str
    nvlink_bandwidth: float  # GB/s
    tdp: float  # Watts

    @property
    def arithmetic_intensity_ridge(self) -> float:
        """
        Calculate the ridge point of the roofline model.

        Ridge point = compute_peak / bandwidth
        (FLOPs/byte where compute-bound vs memory-bound transition occurs)

        Returns:
            Arithmetic intensity at ridge point in FLOPs/byte
        """
        return (self.compute_peak * 1e12) / (self.bandwidth * 1e9)


# Modern GPU specifications based on NVIDIA datasheets
GPU_SPECS: Dict[str, GPUSpec] = {
    "H100": GPUSpec(
        model="H100",
        memory_capacity=80.0,  # GB (HBM3)
        bandwidth=3350.0,  # GB/s (HBM3)
        compute_peak=989.0,  # TFLOPS (FP16 with Tensor Cores, sparsity enabled)
        hbm_speed="HBM3",
        interconnect="NVLink 4.0",
        nvlink_bandwidth=900.0,  # GB/s per GPU (aggregate)
        tdp=700.0,  # Watts (SXM5)
    ),
    "H100-96GB": GPUSpec(
        model="H100-96GB",
        memory_capacity=96.0,  # GB (HBM3)
        bandwidth=3350.0,  # GB/s
        compute_peak=989.0,  # TFLOPS
        hbm_speed="HBM3",
        interconnect="NVLink 4.0",
        nvlink_bandwidth=900.0,
        tdp=700.0,
    ),
    "A100": GPUSpec(
        model="A100",
        memory_capacity=80.0,  # GB (HBM2e)
        bandwidth=2039.0,  # GB/s
        compute_peak=312.0,  # TFLOPS (FP16 with Tensor Cores)
        hbm_speed="HBM2e",
        interconnect="NVLink 3.0",
        nvlink_bandwidth=600.0,  # GB/s per GPU
        tdp=400.0,  # Watts
    ),
    "A100-40GB": GPUSpec(
        model="A100-40GB",
        memory_capacity=40.0,  # GB (HBM2e)
        bandwidth=1555.0,  # GB/s
        compute_peak=312.0,  # TFLOPS
        hbm_speed="HBM2e",
        interconnect="NVLink 3.0",
        nvlink_bandwidth=600.0,
        tdp=250.0,
    ),
    "A30": GPUSpec(
        model="A30",
        memory_capacity=24.0,  # GB (HBM2e)
        bandwidth=933.0,  # GB/s
        compute_peak=66.0,  # TFLOPS (FP16 with Tensor Cores)
        hbm_speed="HBM2e",
        interconnect="NVLink",
        nvlink_bandwidth=300.0,  # GB/s (per direction)
        tdp=165.0,
    ),
    "L40": GPUSpec(
        model="L40",
        memory_capacity=48.0,  # GB (GDDR6)
        bandwidth=864.0,  # GB/s
        compute_peak=181.0,  # TFLOPS (FP16 with Tensor Cores)
        hbm_speed="GDDR6",
        interconnect="PCIe 4.0",
        nvlink_bandwidth=32.0,  # GB/s (PCIe 4.0 x16)
        tdp=300.0,
    ),
    "L4": GPUSpec(
        model="L4",
        memory_capacity=24.0,  # GB (GDDR6)
        bandwidth=300.0,  # GB/s
        compute_peak=39.0,  # TFLOPS (FP16 with Tensor Cores)
        hbm_speed="GDDR6",
        interconnect="PCIe 4.0",
        nvlink_bandwidth=32.0,
        tdp=72.0,
    ),
}


def get_gpu_spec(model: str) -> Optional[GPUSpec]:
    """
    Get GPU specification by model name.

    Args:
        model: GPU model name (e.g., "H100", "A100")

    Returns:
        GPUSpec object if found, None otherwise
    """
    return GPU_SPECS.get(model)


def get_all_gpu_models() -> list[str]:
    """Get list of all available GPU models."""
    return list(GPU_SPECS.keys())


def get_gpus_by_min_memory(min_memory_gb: float) -> list[str]:
    """
    Get GPU models with at least the specified memory capacity.

    Args:
        min_memory_gb: Minimum memory capacity in GB

    Returns:
        List of GPU model names meeting the memory requirement
    """
    return [
        model for model, spec in GPU_SPECS.items()
        if spec.memory_capacity >= min_memory_gb
    ]


def get_gpus_by_compute_peak(min_tflops: float) -> list[str]:
    """
    Get GPU models with at least the specified compute performance.

    Args:
        min_tflops: Minimum compute peak in TFLOPS

    Returns:
        List of GPU model names meeting the compute requirement
    """
    return [
        model for model, spec in GPU_SPECS.items()
        if spec.compute_peak >= min_tflops
    ]


# Default GPU for LLM inference (if not specified)
DEFAULT_LLM_GPU = "H100"

# Cluster configurations
CLUSTER_CONFIGS = {
    "h100_8gpu": {
        "gpus": ["H100"] * 8,
        "description": "8x H100 80GB GPUs with NVLink"
    },
    "h100_16gpu": {
        "gpus": ["H100"] * 16,
        "description": "16x H100 80GB GPUs with NVLink"
    },
    "a100_8gpu": {
        "gpus": ["A100"] * 8,
        "description": "8x A100 80GB GPUs with NVLink"
    },
    "mixed_8gpu": {
        "gpus": ["H100", "H100", "H100", "H100", "A100", "A100", "A30", "L40"],
        "description": "Mixed cluster: 4x H100, 3x A100, 1x A30, 1x L40"
    },
}


def get_cluster_config(cluster_name: str) -> Optional[Dict]:
    """
    Get cluster configuration by name.

    Args:
        cluster_name: Name of the cluster configuration

    Returns:
        Dictionary with 'gpus' list and 'description', or None if not found
    """
    return CLUSTER_CONFIGS.get(cluster_name)


if __name__ == "__main__":
    # Print all available GPU specs
    print("Available GPU Specifications:")
    print("-" * 80)
    for model, spec in GPU_SPECS.items():
        print(f"{model:15} | {spec.memory_capacity:6}GB | {spec.bandwidth:7}GB/s | "
              f"{spec.compute_peak:6.1f} TFLOPS | Ridge: {spec.arithmetic_intensity_ridge:.1f} FLOPs/byte")
    print("-" * 80)

    # Print cluster configurations
    print("\nCluster Configurations:")
    for name, config in CLUSTER_CONFIGS.items():
        print(f"  {name}: {config['description']}")
