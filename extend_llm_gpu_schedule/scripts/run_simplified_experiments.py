#!/usr/bin/env python3
"""
Run simplified LLM GPU scheduling experiments.

This script uses the discrete-time simplified simulator that actually
demonstrates differences between scheduling algorithms.

Key differences from token-based simulator:
- Discrete time steps (dt)
- Tasks complete first, freeing GPUs
- THEN scheduler chooses from waiting queue
- Queue prioritization actually matters!
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List

# Add project root to path FIRST (needed for config imports)
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir.parent))

from extend_llm_gpu_schedule.simulation.simplified_llm_simulator import (
    SimplifiedLLMSimulator,
    SimConfig,
    run_simplified_experiment,
)
from extend_llm_gpu_schedule.models.llm_cluster import create_cluster
from extend_llm_gpu_schedule.models.llm_task import create_prefill_task
from extend_llm_gpu_schedule.algorithms.baseline.llm_fifo import FIFOScheduler
from extend_llm_gpu_schedule.algorithms.baseline.llm_weighted_fifo import WeightedFIFOScheduler
from extend_llm_gpu_schedule.algorithms.baseline.srpt import SRPTScheduler, WSRPTScheduler


def create_discriminating_workload() -> List:
    """
    Create a workload that forces algorithm differences.

    Design:
    - 20 tasks arrive at t=0
    - Cluster has 16 GPUs (can run 8 tasks at once with TP=2)
    - First 8: low weight (1), long duration (2000 tokens)
    - Last 12: high weight (10), short duration (500 tokens)

    Expected behavior:
    - FIFO: Processes in arrival order (T1-T8 first)
    - WeightedFIFO: Prioritizes high weight (T9-T20 first)
    - SRPT: Prioritizes short duration (T9-T20 first)
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


def create_staggered_workload() -> List:
    """
    Create staggered workload where queue builds up.

    Design:
    - First 8 tasks arrive at t=0 and fill all GPUs
    - Next 4 high-weight tasks arrive at t=0.1 and must wait
    - When first batch completes, scheduler chooses from queue
    """
    tasks = []

    # First 8: fill the cluster (will all start at t=0)
    for i in range(8):
        tasks.append(create_prefill_task(
            f"T_fill_{i+1}", "Qwen3", 30, 1.0, 0.0, 1000, 2
        ))

    # Next 4: arrive later and must queue
    for i in range(4):
        tasks.append(create_prefill_task(
            f"T_queue_{i+1}", "Qwen3", 30, 10.0, 0.1, 500, 2
        ))

    return tasks


def print_comparison_table(results: Dict[str, "SimResult"]):
    """Print results comparison table."""
    print("\n" + "=" * 80)
    print("Algorithm Comparison Results")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Weighted Wait':<15} {'Avg Wait':<15} {'Completed':<10}")
    print("-" * 80)

    for name, result in results.items():
        print(f"{name:<20} {result.weighted_waiting_time:<15.3f} "
              f"{result.average_waiting_time:<15.3f} {result.completed_count:<10}")

    print("=" * 80)

    # Calculate improvement ratios
    if "FIFO" in results and "WeightedFIFO" in results:
        fifo_wwt = results["FIFO"].weighted_waiting_time
        weighted_wwt = results["WeightedFIFO"].weighted_waiting_time
        improvement = (fifo_wwt - weighted_wwt) / fifo_wwt * 100
        print(f"\nWeightedFIFO improvement over FIFO: {improvement:.1f}%")

    if "FIFO" in results and "SRPT" in results:
        fifo_avg = results["FIFO"].average_waiting_time
        srpt_avg = results["SRPT"].average_waiting_time
        improvement = (fifo_avg - srpt_avg) / fifo_avg * 100
        print(f"SRPT improvement over FIFO (avg wait): {improvement:.1f}%")


def save_results(results: Dict[str, "SimResult"], output_path: str):
    """Save results to JSON file."""
    output = {}
    for name, result in results.items():
        output[name] = {
            "weighted_waiting_time": result.weighted_waiting_time,
            "average_waiting_time": result.average_waiting_time,
            "makespan": result.makespan,
            "completed_count": result.completed_count,
            "total_count": result.total_count,
            "completion_rate": result.completion_rate,
            "gpu_utilization": result.gpu_utilization,
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run simplified LLM GPU scheduling experiments"
    )
    parser.add_argument(
        "--workload",
        choices=["discriminating", "staggered", "both"],
        default="both",
        help="Workload type to run"
    )
    parser.add_argument(
        "--cluster",
        default="h100_16gpu",
        help="Cluster configuration"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step (seconds)"
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=5.0,
        help="Maximum simulation time (seconds)"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["FIFO", "WeightedFIFO", "SRPT", "WSRPT"],
        choices=["FIFO", "WeightedFIFO", "SRPT", "WSRPT"],
        help="Algorithms to compare"
    )

    args = parser.parse_args()

    # Create cluster
    cluster = create_cluster(args.cluster)
    print(f"Cluster: {args.cluster} ({len(cluster.gpus)} GPUs)")

    # Create schedulers
    scheduler_classes = {
        "FIFO": FIFOScheduler,
        "WeightedFIFO": WeightedFIFOScheduler,
        "SRPT": SRPTScheduler,
        "WSRPT": WSRPTScheduler,
    }

    schedulers = {}
    for name in args.algorithms:
        schedulers[name] = scheduler_classes[name](cluster)

    # Create config
    config = SimConfig(dt=args.dt, max_time=args.max_time)

    # Run experiments
    workloads_to_run = []
    if args.workload == "discriminating" or args.workload == "both":
        workloads_to_run.append(("discriminating", create_discriminating_workload()))
    if args.workload == "staggered" or args.workload == "both":
        workloads_to_run.append(("staggered", create_staggered_workload()))

    all_results = {}

    for workload_name, tasks in workloads_to_run:
        print(f"\n{'='*80}")
        print(f"Running workload: {workload_name}")
        print(f"{'='*80}")
        print(f"Tasks: {len(tasks)}")

        results = run_simplified_experiment(tasks, cluster, schedulers, config)

        for name, result in results.items():
            key = f"{workload_name}_{name}"
            all_results[key] = result

        print_comparison_table(results)

    # Save results if requested
    if args.output:
        save_results(all_results, args.output)


if __name__ == "__main__":
    main()
