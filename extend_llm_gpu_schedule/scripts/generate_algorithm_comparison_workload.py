#!/usr/bin/env python3
"""
Generate workload for algorithm comparison.

This workload is specifically designed to show differences between
FIFO, WeightedFIFO, SRPT, and WSRPT scheduling algorithms.
"""

import argparse
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict
from datetime import datetime


# Model configurations for the test workload
MODEL_CONFIGS = {
    "Llama3-8B": {
        "memory": 16,  # GB
        "tp_degree": 1,
        "weight_range": (1, 3),
        "prefill_tokens": (500, 2000),
        "decode_tokens": (100, 500),
        "probability": 0.25,  # 25% of tasks
    },
    "Qwen3": {
        "memory": 30,  # GB
        "tp_degree": 2,
        "weight_range": (1, 5),
        "prefill_tokens": (1000, 4000),
        "decode_tokens": (200, 1000),
        "probability": 0.35,  # 35% of tasks
    },
    "Llama3-70B": {
        "memory": 70,  # GB
        "tp_degree": 4,
        "weight_range": (3, 8),
        "prefill_tokens": (2000, 6000),
        "decode_tokens": (500, 1500),
        "probability": 0.25,  # 25% of tasks
    },
    "DeepSeek-R1": {
        "memory": 672,  # GB total (84 per GPU x 8)
        "tp_degree": 8,
        "weight_range": (5, 10),
        "prefill_tokens": (3000, 8000),
        "decode_tokens": (1000, 2000),
        "probability": 0.15,  # 15% of tasks (large but rare)
    },
}


def generate_task_id(index: int) -> str:
    """Generate task ID like T1, T2, etc."""
    return f"T{index + 1}"


def generate_single_task(
    task_id: str,
    model_name: str,
    model_config: Dict,
    phase: str,
    min_arrival: float,
    max_arrival: float,
) -> Dict:
    """Generate a single task configuration."""
    # Determine token count based on phase
    if phase == "prefill":
        min_tokens, max_tokens = model_config["prefill_tokens"]
    else:
        min_tokens, max_tokens = model_config["decode_tokens"]

    tokens = random.randint(min_tokens, max_tokens)

    # Weight: vary to show WeightedFIFO vs FIFO difference
    min_weight, max_weight = model_config["weight_range"]
    weight = random.randint(min_weight, max_weight)

    # Arrival time: staggered
    arrival = random.uniform(min_arrival, max_arrival)

    return {
        "Task": task_id,
        "Workload": model_name,
        "Memory": model_config["memory"],
        "Weight": weight,
        "ArrivalTime": round(arrival, 2),
        "Phase": 1 if phase == "prefill" else 0,
        "Tokens": tokens,
    }


def generate_workload(
    num_tasks: int = 80,
    prefill_ratio: float = 0.6,
    arrival_window: float = 100.0,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate workload for algorithm comparison.

    Args:
        num_tasks: Total number of tasks to generate
        prefill_ratio: Ratio of prefill vs decode tasks
        arrival_window: Time window for task arrivals (seconds)
        seed: Random seed for reproducibility

    Returns:
        List of task dictionaries
    """
    random.seed(seed)

    tasks = []
    task_index = 0

    # Determine how many tasks of each model to create
    for model_name, config in MODEL_CONFIGS.items():
        model_count = int(num_tasks * config["probability"])
        print(f"Generating {model_count} tasks for {model_name}")

        for i in range(model_count):
            task_id = generate_task_id(task_index)
            task_index += 1

            # Determine phase (prefill or decode)
            phase = "prefill" if random.random() < prefill_ratio else "decode"

            task = generate_single_task(
                task_id,
                model_name,
                config,
                phase,
                min_arrival=0,
                max_arrival=arrival_window,
            )
            tasks.append(task)

    # Fill remaining tasks if rounding caused shortfall
    while len(tasks) < num_tasks:
        task_id = generate_task_id(task_index)
        task_index += 1

        # Pick random model
        model_name = random.choice(list(MODEL_CONFIGS.keys()))
        config = MODEL_CONFIGS[model_name]
        phase = "prefill" if random.random() < prefill_ratio else "decode"

        task = generate_single_task(
            task_id, model_name, config, phase, 0, arrival_window
        )
        tasks.append(task)

    # Sort by arrival time (for display)
    tasks.sort(key=lambda t: t["ArrivalTime"])

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
    print("Workload Summary")
    print("=" * 70)

    print(f"\nTotal tasks: {len(tasks)}")

    # Model distribution
    print("\nModel distribution:")
    model_counts = df["Workload"].value_counts()
    for model, count in model_counts.items():
        pct = count / len(tasks) * 100
        print(f"  {model:15}: {count:3} ({pct:5.1f}%)")

    # Phase distribution
    print("\nPhase distribution:")
    prefill_count = (df["Phase"] == 1).sum()
    decode_count = (df["Phase"] == 0).sum()
    print(f"  Prefill: {prefill_count:3} ({prefill_count/len(tasks)*100:5.1f}%)")
    print(f"  Decode:  {decode_count:3} ({decode_count/len(tasks)*100:5.1f}%)")

    # Weight distribution
    print("\nWeight distribution:")
    print(f"  Min: {df['Weight'].min()}")
    print(f"  Max: {df['Weight'].max()}")
    print(f"  Mean: {df['Weight'].mean():.2f}")

    # Token distribution
    print("\nToken distribution:")
    prefill_tokens = df[df["Phase"] == 1]["Tokens"]
    decode_tokens = df[df["Phase"] == 0]["Tokens"]
    print(f"  Prefill: {prefill_tokens.min():.0f} - {prefill_tokens.max():.0f}")
    print(f"  Decode:  {decode_tokens.min():.0f} - {decode_tokens.max():.0f}")

    # Arrival time distribution
    print("\nArrival time distribution:")
    print(f"  First: {df['ArrivalTime'].min():.2f}s")
    print(f"  Last: {df['ArrivalTime'].max():.2f}s")

    # Estimated GPU requirements
    print("\nEstimated GPU requirements:")
    try:
        from extend_llm_gpu_schedule.config.llm_model_specs import LLM_MODEL_SPECS

        for model_name in df["Workload"].unique():
            model_df = df[df["Workload"] == model_name]
            if model_name in LLM_MODEL_SPECS:
                spec = LLM_MODEL_SPECS[model_name]
                tp = spec.recommended_tp
                print(f"  {model_name:15}: TP{tp} ({tp} GPUs per task)")
    except ImportError:
        # Running from different directory context
        pass

    print("\n" + "=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate workload for LLM scheduling algorithm comparison"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=80,
        help="Number of tasks to generate (default: 80)",
    )
    parser.add_argument(
        "--prefill-ratio",
        type=float,
        default=0.6,
        help="Ratio of prefill tasks (default: 0.6)",
    )
    parser.add_argument(
        "--arrival-window",
        type=float,
        default=100.0,
        help="Time window for task arrivals in seconds (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/llm_workload/algo_comparison.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    print("LLM Scheduling Algorithm Comparison Workload Generator")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Tasks: {args.num_tasks}")
    print(f"  Prefill ratio: {args.prefill_ratio:.1%}")
    print(f"  Arrival window: {args.arrival_window}s")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate workload
    tasks = generate_workload(
        num_tasks=args.num_tasks,
        prefill_ratio=args.prefill_ratio,
        arrival_window=args.arrival_window,
        seed=args.seed,
    )

    # Save workload
    save_workload(tasks, args.output)

    # Print summary
    print_workload_summary(tasks)

    print("\nRecommended cluster configurations for testing:")
    print("  1. Small:  h100_16gpu  (16x H100 80GB)")
    print("  2. Medium: h100_32gpu  (32x H100 80GB)")
    print("  3. Large:  mixed_32gpu (mix of H100, A100, A30)")


if __name__ == "__main__":
    main()
