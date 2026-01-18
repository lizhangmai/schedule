#!/usr/bin/env python3
"""
Generate synthetic LLM inference workloads for scheduling simulation.

Creates CSV files with LLM tasks including prefill/decode phases,
token counts, and model specifications.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Dict, Literal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from extend_llm_gpu_schedule.config.llm_model_specs import LLM_MODEL_SPECS, get_model_spec
from extend_llm_gpu_schedule.models.llm_task import Phase, create_prefill_task, create_decode_task
from extend_llm_gpu_schedule.utils.llm_data_loader import save_llm_tasks_to_csv, print_dataset_summary


# Available models for workload generation
AVAILABLE_MODELS = [
    "DeepSeek-R1",
    "Qwen3",
    "GLM4.7",
    "Llama3-70B",
    "Llama3-8B",
    "Mistral-7B",
]


# Load level configurations
LOAD_LEVELS = {
    "low": {
        "num_tasks": 100,
        "arrival_range": (0, 100),  # seconds
        "token_range_prefill": (512, 4096),
        "token_range_decode": (128, 1024),
        "weight_range": (1, 5),
        "description": "Low load: Sparse arrivals, small requests",
    },
    "medium": {
        "num_tasks": 500,
        "arrival_range": (0, 50),
        "token_range_prefill": (512, 8192),
        "token_range_decode": (128, 2048),
        "weight_range": (1, 10),
        "description": "Medium load: Moderate arrival rate",
    },
    "high": {
        "num_tasks": 1000,
        "arrival_range": (0, 30),
        "token_range_prefill": (1024, 16384),
        "token_range_decode": (256, 4096),
        "weight_range": (1, 10),
        "description": "High load: Dense arrivals, large requests",
    },
    "extreme": {
        "num_tasks": 2000,
        "arrival_range": (0, 20),
        "token_range_prefill": (2048, 32768),
        "token_range_decode": (512, 8192),
        "weight_range": (1, 10),
        "description": "Extreme load: Very dense, very large requests",
    },
}


def generate_llm_workload(
    num_tasks: int,
    seed: int = 42,
    models: List[str] = None,
    arrival_range: tuple = (0, 50),
    token_range_prefill: tuple = (512, 8192),
    token_range_decode: tuple = (128, 2048),
    weight_range: tuple = (1, 10),
    phase_distribution: tuple = (0.7, 0.3),  # (prefill, decode)
    separate_phases: bool = True
) -> List[Dict]:
    """
    Generate LLM workload tasks.

    Args:
        num_tasks: Number of tasks to generate
        seed: Random seed
        models: List of models to sample from
        arrival_range: Task arrival time range (min, max)
        token_range_prefill: Prefill token range (min, max)
        token_range_decode: Decode token range (min, max)
        weight_range: Task weight range (min, max)
        phase_distribution: Probability distribution for phases (prefill, decode)
        separate_phases: Generate separate prefill/decode tasks (True) or combined (False)

    Returns:
        List of task dictionaries
    """
    random.seed(seed)

    if models is None:
        models = AVAILABLE_MODELS

    # Filter to available models
    models = [m for m in models if m in LLM_MODEL_SPECS]
    if not models:
        raise ValueError(f"No valid models specified. Available: {AVAILABLE_MODELS}")

    tasks = []
    arrival_times = sorted(random.uniform(*arrival_range) for _ in range(num_tasks))

    for i, arrival_time in enumerate(arrival_times, 1):
        model = random.choice(models)
        model_spec = get_model_spec(model)

        if model_spec is None:
            continue

        # Determine phase
        phase_prefill_prob = phase_distribution[0]
        is_prefill = random.random() < phase_prefill_prob

        # Generate token count
        if is_prefill:
            tokens = random.randint(*token_range_prefill)
            phase = Phase.PREFILL
        else:
            tokens = random.randint(*token_range_decode)
            phase = Phase.DECODE

        # Get TP degree
        tp_degree = model_spec.recommended_tp
        memory_per_gpu = model_spec.get_memory_for_tp(tp_degree)

        # Total memory
        total_memory = memory_per_gpu * tp_degree if memory_per_gpu else 100

        # Weight (priority)
        weight = random.randint(*weight_range)

        tasks.append({
            "task_id": f"T{i}",
            "model_name": model,
            "memory": total_memory,
            "weight": weight,
            "arrival_time": round(arrival_time, 4),
            "phase": phase,
            "tokens": tokens,
            "tp_degree": tp_degree,
        })

    return tasks


def create_tasks_from_dicts(task_dicts: List[Dict]) -> List:
    """Convert task dictionaries to LLMTask objects."""
    from extend_llm_gpu_schedule.models.llm_task import create_prefill_task, create_decode_task

    tasks = []
    for task_dict in task_dicts:
        if task_dict["phase"] == Phase.PREFILL:
            task = create_prefill_task(
                task_id=task_dict["task_id"],
                model_name=task_dict["model_name"],
                memory=task_dict["memory"],
                weight=task_dict["weight"],
                arrival_time=task_dict["arrival_time"],
                tokens=task_dict["tokens"],
                tp_degree=task_dict["tp_degree"]
            )
        else:
            task = create_decode_task(
                task_id=task_dict["task_id"],
                model_name=task_dict["model_name"],
                memory=task_dict["memory"],
                weight=task_dict["weight"],
                arrival_time=task_dict["arrival_time"],
                tokens=task_dict["tokens"],
                tp_degree=task_dict["tp_degree"]
            )
        tasks.append(task)

    return tasks


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate LLM workload datasets for scheduling simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-id", type=int, help="Task dataset ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-tasks",
        type=int,
        help="Number of tasks (overrides load-level)"
    )
    parser.add_argument(
        "--load-level",
        type=str,
        choices=["low", "medium", "high", "extreme"],
        default="medium",
        help="Load level preset"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=AVAILABLE_MODELS,
        help=f"Models to include in workload. Available: {AVAILABLE_MODELS}"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/llm_workload",
        help="Output directory for generated CSV files"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print dataset summary after generation"
    )

    args = parser.parse_args()

    # Load configuration
    load_config = LOAD_LEVELS[args.load_level]

    # Override num_tasks if specified
    num_tasks = args.num_tasks if args.num_tasks else load_config["num_tasks"]

    # Generate workload
    print(f"Generating LLM workload with {args.load_level} load level...")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Tasks: {num_tasks}")

    task_dicts = generate_llm_workload(
        num_tasks=num_tasks,
        seed=args.seed,
        models=args.models,
        arrival_range=load_config["arrival_range"],
        token_range_prefill=load_config["token_range_prefill"],
        token_range_decode=load_config["token_range_decode"],
        weight_range=load_config["weight_range"],
    )

    # Convert to LLMTask objects
    tasks = create_tasks_from_dicts(task_dicts)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    task_id = args.task_id if args.task_id else 1
    output_path = output_dir / f"llm_task{task_id}.csv"

    save_llm_tasks_to_csv(tasks, str(output_path))

    # Print summary
    if args.summary:
        print_dataset_summary(tasks)

    print(f"\nLoad level: {args.load_level}")
    print(f"Description: {load_config['description']}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
