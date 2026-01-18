#!/usr/bin/env python3
"""
Run LLM scheduling experiments.

Executes scheduling simulations with different algorithms and workloads,
calculates metrics, and saves results for analysis.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from extend_llm_gpu_schedule.models.llm_task import Phase
    from extend_llm_gpu_schedule.models.llm_cluster import CLUSTER_PRESETS
    from extend_llm_gpu_schedule.utils.llm_data_loader import load_llm_tasks_from_csv, validate_llm_tasks
    from extend_llm_gpu_schedule.metrics.llm_metrics import LLMMetricsCalculator
except ImportError:
    # Fallback to relative imports
    from models.llm_task import Phase
    from models.llm_cluster import CLUSTER_PRESETS
    from utils.llm_data_loader import load_llm_tasks_from_csv, validate_llm_tasks
    from metrics.llm_metrics import LLMMetricsCalculator


def create_scheduler(algorithm_name: str, cluster):
    """Create a scheduler instance."""
    # Import here to avoid circular dependencies
    from extend_llm_gpu_schedule.algorithms.base_llm_scheduler import BaseLLMScheduler
    from extend_llm_gpu_schedule.algorithms.baseline.llm_fifo import FIFOScheduler
    from extend_llm_gpu_schedule.algorithms.baseline.llm_weighted_fifo import (
        WeightedFIFOScheduler,
        StrictWeightedFIFOScheduler
    )
    from extend_llm_gpu_schedule.algorithms.baseline.srpt import SRPTScheduler, WSRPTScheduler
    from extend_llm_gpu_schedule.simulation.llm_simulator import LLMSimulator
    from extend_llm_gpu_schedule.performance.roofline_model import create_roofline_calculator
    from extend_llm_gpu_schedule.performance.throughput_estimator import create_throughput_estimator
    from extend_llm_gpu_schedule.models.llm_cluster import create_llm_cluster_from_configs

    schedulers = {
        "FIFO": lambda c: FIFOScheduler(c),
        "WeightedFIFO": lambda c: WeightedFIFOScheduler(c),
        "StrictWeightedFIFO": lambda c: StrictWeightedFIFOScheduler(c),
        "SRPT": lambda c: SRPTScheduler(c),
        "WSRPT": lambda c: WSRPTScheduler(c),
    }

    scheduler_fn = schedulers.get(algorithm_name)
    if scheduler_fn is None:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(schedulers.keys())}")

    return scheduler_fn(cluster)


def run_single_experiment(
    tasks,
    cluster,
    algorithm_name: str,
    verbose: bool = False
) -> Dict:
    """
    Run a single experiment with one algorithm.

    Args:
        tasks: List of LLM tasks
        cluster: LLM cluster
        algorithm_name: Name of scheduling algorithm
        verbose: Print progress information

    Returns:
        Dictionary with experiment results and scheduled tasks
    """
    if verbose:
        print(f"  Running {algorithm_name}...")

    # Import here to avoid circular dependencies
    from extend_llm_gpu_schedule.simulation.llm_simulator import LLMSimulator
    from extend_llm_gpu_schedule.performance.roofline_model import create_roofline_calculator
    from extend_llm_gpu_schedule.performance.throughput_estimator import create_throughput_estimator
    import copy

    # Create scheduler
    scheduler = create_scheduler(algorithm_name, cluster)

    # Create simulator
    calculator = create_roofline_calculator()
    estimator = create_throughput_estimator(calculator)
    simulator = LLMSimulator(cluster, estimator)

    # Deep copy tasks to avoid state contamination
    tasks_copy = copy.deepcopy(tasks)

    # Reset state
    for task in tasks_copy:
        task.reset()
    cluster.reset()

    # Run simulation
    try:
        result = simulator.run(tasks_copy, scheduler, max_time=1e6)

        # Calculate metrics
        metrics = LLMMetricsCalculator.calculate(
            result.tasks,
            cluster,
            result.makespan,
            batching_stats=simulator.batching_manager.get_statistics()
        )

        return {
            "algorithm": algorithm_name,
            "success": True,
            "metrics": metrics.to_dict(),
            "state": {
                "completed_tasks": metrics.completed_tasks,
                "total_tasks": metrics.total_tasks,
                "makespan": metrics.makespan,
            },
            "tasks": result.tasks,  # Include scheduled tasks for visualization
        }
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return {
            "algorithm": algorithm_name,
            "success": False,
            "error": str(e),
            "tasks": [],
        }


def run_experiment_suite(
    workload_file: str,
    cluster_config: str,
    algorithms: List[str],
    verbose: bool = False
) -> Tuple[List[Dict], object]:
    """
    Run experiments with multiple algorithms on a workload.

    Args:
        workload_file: Path to workload CSV file
        cluster_config: Cluster configuration name
        algorithms: List of algorithm names to test
        verbose: Print progress

    Returns:
        Tuple of (list of experiment results, cluster object)
    """
    # Import here to avoid circular dependencies
    from extend_llm_gpu_schedule.models.llm_cluster import create_cluster_from_cluster_config

    # Load workload
    if verbose:
        print(f"Loading workload from {workload_file}...")

    tasks = load_llm_tasks_from_csv(workload_file)
    validation = validate_llm_tasks(tasks)

    if not validation["valid"]:
        print("Validation errors:")
        for error in validation["errors"]:
            print(f"  ERROR: {error}")
        raise ValueError("Invalid workload dataset")

    if verbose:
        print(f"  Loaded {len(tasks)} tasks")
        stats = validation["statistics"]
        print(f"  Models: {list(stats['models'].keys())}")
        print(f"  Phases: Prefill={stats['phases'][Phase.PREFILL]}, Decode={stats['phases'][Phase.DECODE]}")

    # Create cluster
    if verbose:
        print(f"\nCreating cluster: {cluster_config}")

    cluster = create_cluster_from_cluster_config(cluster_config)
    if cluster is None:
        raise ValueError(f"Unknown cluster config: {cluster_config}")

    if verbose:
        print(f"  GPUs: {cluster.gpu_count}")
        print(f"  Distribution: {cluster.get_model_distribution()}")

    # Run experiments for each algorithm
    results = []
    for algo in algorithms:
        result = run_single_experiment(tasks, cluster, algo, verbose)
        results.append(result)

    return results, cluster


def save_results(
    results: List[Dict],
    output_dir: str,
    workload_name: str,
    cluster_config: str,
    cluster,
    generate_plots: bool = False
) -> None:
    """
    Save experiment results to files.

    Args:
        results: List of experiment results
        output_dir: Output directory path
        workload_name: Name of workload
        cluster_config: Cluster configuration name
        cluster: LLM cluster object
        generate_plots: Whether to generate visualization plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = f"{workload_name}_{cluster_config}"

    # Save comparison CSV
    import pandas as pd

    comparison_data = []
    for result in results:
        if result["success"]:
            row = {
                "Algorithm": result["algorithm"],
                **result["metrics"]
            }
            comparison_data.append(row)

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        csv_path = output_path / f"{base_name}_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved comparison to: {csv_path}")

        # Print comparison table
        print("\n" + "=" * 100)
        print("Algorithm Comparison:")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)

    # Save detailed JSON
    json_path = output_path / f"{base_name}_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "workload": workload_name,
            "cluster": cluster_config,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2, default=str)
    print(f"Saved detailed results to: {json_path}")

    # Generate visualization plots if requested
    if generate_plots:
        from extend_llm_gpu_schedule.visualization.llm_plots import LLMPlotGenerator

        # Prepare data for visualization
        results_for_plot = {}
        tasks_per_algorithm = {}

        for result in results:
            if result["success"]:
                results_for_plot[result["algorithm"]] = result
                if "tasks" in result:
                    tasks_per_algorithm[result["algorithm"]] = result["tasks"]

        if results_for_plot:
            LLMPlotGenerator.generate_all_plots(
                results_for_plot,
                tasks_per_algorithm,
                cluster,
                output_dir,
                base_name,
            )


def print_summary(results: List[Dict]) -> None:
    """Print summary of experiment results."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*80}")
    print("Experiment Summary")
    print(f"{'='*80}")

    if successful:
        # Find best algorithm by primary objective (weighted waiting time)
        best = min(successful, key=lambda r: r["metrics"]["weighted_waiting_time"])
        worst = max(successful, key=lambda r: r["metrics"]["weighted_waiting_time"])

        print(f"\nTotal experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        print(f"\nBest algorithm (by weighted waiting time):")
        print(f"  {best['algorithm']}: {best['metrics']['weighted_waiting_time']:.2f}")

        print(f"\nWorst algorithm (by weighted waiting time):")
        print(f"  {worst['algorithm']}: {worst['metrics']['weighted_waiting_time']:.2f}")

        improvement = (
            (worst['metrics']['weighted_waiting_time'] - best['metrics']['weighted_waiting_time']) /
            worst['metrics']['weighted_waiting_time'] * 100
        )
        print(f"\nImprovement: {improvement:.1f}%")

    if failed:
        print(f"\nFailed algorithms:")
        for result in failed:
            print(f"  {result['algorithm']}: {result.get('error', 'Unknown error')}")

    print(f"\n{'='*80}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run LLM GPU scheduling experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to workload CSV file or dataset name (e.g., data/llm_workload/llm_task1.csv)"
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="h100_8gpu",
        help="Cluster configuration (default: h100_8gpu)"
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["FIFO", "WeightedFIFO", "SRPT", "WSRPT"],
        help="Algorithms to test (default: FIFO WeightedFIFO SRPT WSRPT)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/llm_experiments",
        help="Output directory for results (default: results/llm_experiments)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--list-clusters",
        action="store_true",
        help="List available cluster configurations"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots"
    )

    args = parser.parse_args()

    # List available clusters
    if args.list_clusters:
        print("Available Cluster Configurations:")
        print("-" * 60)
        for name, config in CLUSTER_PRESETS.items():
            print(f"  {name:20}: {config['description']}")
        print("\nYou can also specify GPU counts like: h100_4gpu, a100_16gpu")
        return

    # Validate dataset is provided
    if not args.dataset:
        print("Error: --dataset is required (unless using --list-clusters)")
        parser.print_help()
        sys.exit(1)

    # Validate dataset file
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        # Try default path
        default_path = Path("data/llm_workload") / args.dataset
        if default_path.exists():
            dataset_path = default_path
        else:
            print(f"Error: Dataset file not found: {args.dataset}")
            print(f"  Tried: {dataset_path}")
            print(f"  Tried: {default_path}")
            sys.exit(1)

    workload_name = dataset_path.stem

    print("=" * 80)
    print("LLM GPU Scheduling Experiments")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Cluster: {args.cluster}")
    print(f"Algorithms: {', '.join(args.algorithms)}")
    print(f"Output: {args.output_dir}")
    print(f"Generate Plots: {args.plot}")
    print("=" * 80)

    # Run experiments
    results, cluster = run_experiment_suite(
        workload_file=str(dataset_path),
        cluster_config=args.cluster,
        algorithms=args.algorithms,
        verbose=args.verbose
    )

    # Save results
    save_results(
        results,
        args.output_dir,
        workload_name,
        args.cluster,
        cluster,
        generate_plots=args.plot
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
