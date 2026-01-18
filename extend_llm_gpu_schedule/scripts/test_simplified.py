# Set up paths FIRST (before any other imports)
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

#!/usr/bin/env python3
"""
Simple standalone test of simplified LLM GPU scheduling simulator.

This script demonstrates algorithm differences without complex import dependencies.
"""

# Import what we need directly
from extend_llm_gpu_schedule.simulation.simplified_llm_simulator import (
    SimplifiedLLMSimulator,
    SimConfig,
    run_simplified_experiment,
)
from extend_llm_gpu_schedule.models.llm_cluster import create_cluster_from_cluster_config
from extend_llm_gpu_schedule.models.llm_task import create_prefill_task
from extend_llm_gpu_schedule.algorithms.baseline.llm_fifo import FIFOScheduler
from extend_llm_gpu_schedule.algorithms.baseline.llm_weighted_fifo import WeightedFIFOScheduler
from extend_llm_gpu_schedule.algorithms.baseline.srpt import SRPTScheduler
from extend_llm_gpu_schedule.algorithms.baseline.srpt import WSRPTScheduler


def create_discriminating_workload():
    """
    Create workload that shows algorithm differences.

    Design:
    - 20 tasks arrive at t=0
    - Cluster has 16 GPUs (8 concurrent tasks with TP=2)
    - First 8: LOW weight (1), LONG duration (2000 tokens)
    - Last 12: HIGH weight (10), SHORT duration (500 tokens)

    Expected:
    - FIFO: T1-T8 first, then T9-T20
    - WeightedFIFO: T9-T20 first (high weight)
    - SRPT: T9-T20 first (short duration)
    """
    tasks = []

    # First 8: LOW weight, LONG duration
    for i in range(8):
        tasks.append(create_prefill_task(
            f"T_low_{i+1}", "Qwen3", 30, 1.0, 0.0, 2000, 2
        ))

    # Next 12: HIGH weight, SHORT duration
    for i in range(12):
        tasks.append(create_prefill_task(
            f"T_high_{i+1}", "Qwen3", 30, 10.0, 0.0, 500, 2
        ))

    return tasks


def main():
    print("=" * 80)
    print("Simplified LLM GPU Scheduling Simulator Test")
    print("=" * 80)

    # Create cluster
    cluster = create_cluster_from_cluster_config("h100_16gpu")
    print(f"\nCluster: h100_16gpu ({len(cluster.gpus)} GPUs)")

    # Create workload
    tasks = create_discriminating_workload()
    print(f"Tasks: {len(tasks)}")
    print("  - T_low_1 to T_low_8: weight=1, tokens=2000 (long)")
    print("  - T_high_1 to T_high_12: weight=10, tokens=500 (short)")

    # Create schedulers
    schedulers = {
        "FIFO": FIFOScheduler(cluster),
        "WeightedFIFO": WeightedFIFOScheduler(cluster),
        "SRPT": SRPTScheduler(cluster),
        "WSRPT": WSRPTScheduler(cluster),
    }

    # Create config
    config = SimConfig(dt=0.01, max_time=5.0)

    # Run experiments
    print("\n" + "=" * 80)
    print("Running experiments...")
    print("=" * 80)

    results = run_simplified_experiment(tasks, cluster, schedulers, config)

    # Print results
    print("\n" + "=" * 80)
    print("Algorithm Comparison Results")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Weighted Wait':<15} {'Avg Wait':<15} {'Completed':<10}")
    print("-" * 80)

    for name, result in results.items():
        print(f"{name:<20} {result.weighted_waiting_time:<15.3f} "
              f"{result.average_waiting_time:<15.3f} {result.completed_count:<10}")

    print("=" * 80)

    # Show differences
    fifo_wwt = results["FIFO"].weighted_waiting_time
    weighted_wwt = results["WeightedFIFO"].weighted_waiting_time
    srpt_avg = results["SRPT"].average_waiting_time
    fifo_avg = results["FIFO"].average_waiting_time

    print(f"\nWeightedFIFO vs FIFO (weighted waiting time):")
    print(f"  FIFO: {fifo_wwt:.3f}")
    print(f"  WeightedFIFO: {weighted_wwt:.3f}")
    if fifo_wwt > 0:
        improvement = (fifo_wwt - weighted_wwt) / fifo_wwt * 100
        print(f"  Improvement: {improvement:.1f}%")

    print(f"\nSRPT vs FIFO (average waiting time):")
    print(f"  FIFO: {fifo_avg:.3f}")
    print(f"  SRPT: {srpt_avg:.3f}")
    if fifo_avg > 0:
        improvement = (fifo_avg - srpt_avg) / fifo_avg * 100
        print(f"  Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
