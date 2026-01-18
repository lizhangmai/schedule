#!/usr/bin/env python3
"""
Generate workload with burst arrivals to show algorithm differences.

This creates scenarios where many tasks arrive at the same time,
forcing schedulers to make different prioritization decisions.
"""

import argparse
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict


# Simplified model configs
MODEL_CONFIGS = {
    "Llama3-8B": {
        "memory": 16,
        "tp_degree": 1,
        "weight_range": (1, 10),
        "tokens": (500, 2000),
    },
    "Qwen3": {
        "memory": 30,
        "tp_degree": 2,
        "weight_range": (1, 10),
        "tokens": (1000, 4000),
    },
    "Llama3-70B": {
        "memory": 70,
        "tp_degree": 4,
        "weight_range": (1, 10),
        "tokens": (2000, 6000),
    },
}


def generate_burst_workload(
    burst_size: int = 20,
    num_bursts: int = 3,
    burst_interval: float = 50.0,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate workload with burst arrivals.

    Args:
        burst_size: Number of tasks in each burst
        num_bursts: Number of bursts
        burst_interval: Time between bursts
        seed: Random seed

    Returns:
        List of task dictionaries
    """
    random.seed(seed)
    tasks = []
    task_index = 0

    models = list(MODEL_CONFIGS.keys())

    for burst_idx in range(num_bursts):
        burst_time = burst_idx * burst_interval

        print(f"Burst {burst_idx + 1} at t={burst_time:.1f}s:")

        for i in range(burst_size):
            task_id = f"T{task_index + 1}"

            # Pick random model
            model_name = random.choice(models)
            config = MODEL_CONFIGS[model_name]

            # Generate task properties
            weight = random.randint(*config["weight_range"])
            tokens = random.randint(*config["tokens"])

            # All tasks arrive at nearly the same time (within 0.1 seconds)
            arrival = burst_time + random.uniform(0, 0.1)

            tasks.append({
                "Task": task_id,
                "Workload": model_name,
                "Memory": config["memory"],
                "Weight": weight,
                "ArrivalTime": round(arrival, 3),
                "Phase": 1,  # Prefill
                "Tokens": tokens,
            })

            task_index += 1

        print(f"  Generated {burst_size} tasks")

    return tasks


def save_workload(tasks: List[Dict], output_path: str) -> None:
    """Save tasks to CSV file."""
    df = pd.DataFrame(tasks)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(tasks)} tasks to {output_path}")


def print_workload_summary(tasks: List[Dict]) -> None:
    """Print summary of generated workload."""
    df = pd.DataFrame(tasks)

    print("\n" + "=" * 70)
    print("Burst Workload Summary")
    print("=" * 70)

    print(f"\nTotal tasks: {len(tasks)}")

    # Group arrivals into bursts
    df_sorted = df.sort_values("ArrivalTime")
    bursts = []
    current_burst = []
    last_time = None

    for _, row in df_sorted.iterrows():
        if last_time is None or row["ArrivalTime"] - last_time > 1.0:
            if current_burst:
                bursts.append(current_burst)
            current_burst = []
        current_burst.append(row)
        last_time = row["ArrivalTime"]

    if current_burst:
        bursts.append(current_burst)

    print(f"\nDetected {len(bursts)} arrival bursts:")
    for i, burst in enumerate(bursts):
        print(f"  Burst {i + 1}: {len(burst)} tasks at t={burst[0]['ArrivalTime']:.2f}s")

    # Model distribution
    print("\nModel distribution:")
    model_counts = df["Workload"].value_counts()
    for model, count in model_counts.items():
        config = MODEL_CONFIGS[model]
        print(f"  {model:15}: {count:3} tasks (TP={config['tp_degree']}, {config['memory']}GB)")

    # Weight distribution
    print("\nWeight distribution:")
    print(f"  Min: {df['Weight'].min()}")
    print(f"  Max: {df['Weight'].max()}")
    print(f"  Mean: {df['Weight'].mean():.2f}")

    # Token distribution
    print("\nToken distribution:")
    print(f"  Min: {df['Tokens'].min()}")
    print(f"  Max: {df['Tokens'].max()}")
    print(f"  Mean: {df['Tokens'].mean():.0f}")

    print("\n" + "=" * 70)
    print("\nThis workload is designed to show differences between algorithms:")
    print("- FIFO schedules by arrival time (within burst)")
    print("- WeightedFIFO schedules by weight (highest first)")
    print("- SRPT schedules by estimated duration (shortest first)")
    print("- WSRPT schedules by weight/duration ratio")
    print("\nRecommended cluster: h100_16gpu (limited capacity to force queueing)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate burst workload for algorithm comparison"
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=20,
        help="Number of tasks per burst (default: 20)",
    )
    parser.add_argument(
        "--num-bursts",
        type=int,
        default=3,
        help="Number of bursts (default: 3)",
    )
    parser.add_argument(
        "--burst-interval",
        type=float,
        default=50.0,
        help="Time between bursts in seconds (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/llm_workload/burst_comparison.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    print("LLM Scheduling Burst Workload Generator")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Burst size: {args.burst_size} tasks")
    print(f"  Number of bursts: {args.num_bursts}")
    print(f"  Burst interval: {args.burst_interval}s")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate workload
    tasks = generate_burst_workload(
        burst_size=args.burst_size,
        num_bursts=args.num_bursts,
        burst_interval=args.burst_interval,
        seed=args.seed,
    )

    # Save workload
    save_workload(tasks, args.output)

    # Print summary
    print_workload_summary(tasks)


if __name__ == "__main__":
    main()
