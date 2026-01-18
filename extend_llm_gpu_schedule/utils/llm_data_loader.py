"""
LLM workload data loader.

Loads and validates LLM tasks from CSV files.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict

from extend_llm_gpu_schedule.models.llm_task import LLMTask, Phase, create_prefill_task, create_decode_task
from extend_llm_gpu_schedule.config.llm_model_specs import LLM_MODEL_SPECS, get_model_spec
from extend_llm_gpu_schedule.config.tp_config import get_recommended_tp


def load_llm_tasks_from_csv(file_path: str) -> List[LLMTask]:
    """
    Load LLM tasks from CSV file.

    Expected CSV format:
    Task,Workload,Memory,Weight,ArrivalTime,Phase,Tokens
    T1,DeepSeek-R1,671,17.98,0.36,1,2048

    Args:
        file_path: Path to CSV file

    Returns:
        List of LLMTask objects
    """
    df = pd.read_csv(file_path)

    tasks = []
    for _, row in df.iterrows():
        # Support both numeric (1/0) and string ("PREFILL"/"DECODE") phase values
        phase_value = row["Phase"]
        if phase_value == 1 or phase_value == "1" or str(phase_value).upper() == "PREFILL":
            phase = Phase.PREFILL
        else:
            phase = Phase.DECODE

        # Get model spec to determine TP degree
        model_spec = get_model_spec(row["Workload"])
        if model_spec is None:
            print(f"Warning: Unknown model '{row['Workload']}' in task {row['Task']}")
            # Use default TP=1
            tp_degree = 1
        else:
            tp_degree = model_spec.recommended_tp

        if phase == Phase.PREFILL:
            task = create_prefill_task(
                task_id=row["Task"],
                model_name=row["Workload"],
                memory=row["Memory"],
                weight=row["Weight"],
                arrival_time=row["ArrivalTime"],
                tokens=int(row["Tokens"]),
                tp_degree=tp_degree
            )
        else:
            task = create_decode_task(
                task_id=row["Task"],
                model_name=row["Workload"],
                memory=row["Memory"],
                weight=row["Weight"],
                arrival_time=row["ArrivalTime"],
                tokens=int(row["Tokens"]),
                tp_degree=tp_degree
            )

        tasks.append(task)

    return tasks


def save_llm_tasks_to_csv(tasks: List[LLMTask], file_path: str) -> None:
    """
    Save LLM tasks to CSV file.

    Args:
        tasks: List of LLMTask objects
        file_path: Path to output CSV file
    """
    data = []
    for task in tasks:
        # Save phase as numeric (1 for PREFILL, 0 for DECODE) for compatibility
        phase_value = 1 if task.phase == Phase.PREFILL else 0
        data.append({
            "Task": task.task_id,
            "Workload": task.model_name,
            "Memory": task.memory,
            "Weight": task.weight,
            "ArrivalTime": task.arrival_time,
            "Phase": phase_value,
            "Tokens": task.tokens,
        })

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Saved {len(tasks)} LLM tasks to {file_path}")


def validate_llm_tasks(tasks: List[LLMTask]) -> Dict[str, any]:
    """
    Validate LLM task dataset.

    Args:
        tasks: List of LLM tasks

    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []
    stats = {
        "total_tasks": len(tasks),
        "models": {},
        "phases": {Phase.PREFILL: 0, Phase.DECODE: 0},
        "token_ranges": {},
    }

    task_ids = set()

    for task in tasks:
        # Check task ID uniqueness
        if task.task_id in task_ids:
            errors.append(f"Duplicate task ID: {task.task_id}")
        task_ids.add(task.task_id)

        # Check model exists
        if task.model_name not in LLM_MODEL_SPECS:
            errors.append(f"Unknown model '{task.model_name}' in task {task.task_id}")
        else:
            # Track model stats
            if task.model_name not in stats["models"]:
                stats["models"][task.model_name] = 0
            stats["models"][task.model_name] += 1

            # Check memory requirement
            model_spec = get_model_spec(task.model_name)
            if model_spec:
                mem_per_gpu = model_spec.get_memory_for_tp(task.tp_degree)
                if mem_per_gpu and task.memory > model_spec.parameters * 3:
                    warnings.append(
                        f"Memory requirement seems high for task {task.task_id}: "
                        f"{task.memory}GB for {task.model_name}"
                    )

        # Track phases
        stats["phases"][task.phase] += 1

        # Track token ranges
        if task.phase not in stats["token_ranges"]:
            stats["token_ranges"][task.phase] = {"min": float('inf'), "max": 0}
        stats["token_ranges"][task.phase]["min"] = min(
            stats["token_ranges"][task.phase]["min"], task.tokens
        )
        stats["token_ranges"][task.phase]["max"] = max(
            stats["token_ranges"][task.phase]["max"], task.tokens
        )

        # Check for negative values
        if task.weight < 0:
            errors.append(f"Negative weight in task {task.task_id}")
        if task.arrival_time < 0:
            errors.append(f"Negative arrival time in task {task.task_id}")
        if task.tokens < 0:
            errors.append(f"Negative token count in task {task.task_id}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "statistics": stats,
    }


def print_dataset_summary(tasks: List[LLMTask]) -> None:
    """
    Print summary of LLM task dataset.

    Args:
        tasks: List of LLM tasks
    """
    validation = validate_llm_tasks(tasks)
    stats = validation["statistics"]

    print(f"\n{'='*60}")
    print(f"LLM Workload Summary")
    print(f"{'='*60}")
    print(f"Total tasks: {stats['total_tasks']}")

    print(f"\nModels:")
    for model, count in stats["models"].items():
        print(f"  {model:20}: {count:4} tasks")

    print(f"\nPhases:")
    for phase, count in stats["phases"].items():
        phase_name = "Prefill" if phase == Phase.PREFILL else "Decode"
        print(f"  {phase_name:10}: {count:4} tasks")

    print(f"\nToken ranges:")
    for phase, rng in stats["token_ranges"].items():
        phase_name = "Prefill" if phase == Phase.PREFILL else "Decode"
        print(f"  {phase_name:10}: {rng['min']:.0f} - {rng['max']:.0f} tokens")

    if validation["errors"]:
        print(f"\nErrors ({len(validation['errors'])}):")
        for error in validation["errors"][:5]:  # Show first 5
            print(f"  - {error}")
        if len(validation["errors"]) > 5:
            print(f"  ... and {len(validation['errors']) - 5} more")

    if validation["warnings"]:
        print(f"\nWarnings ({len(validation['warnings'])}):")
        for warning in validation["warnings"][:5]:
            print(f"  - {warning}")
        if len(validation["warnings"]) > 5:
            print(f"  ... and {len(validation['warnings']) - 5} more")

    print(f"{'='*60}\n")


def load_and_validate(file_path: str) -> Optional[List[LLMTask]]:
    """
    Load LLM tasks from CSV and validate.

    Args:
        file_path: Path to CSV file

    Returns:
        List of LLMTask objects if valid, None otherwise
    """
    try:
        tasks = load_llm_tasks_from_csv(file_path)
        validation = validate_llm_tasks(tasks)

        if not validation["valid"]:
            print("Validation errors:")
            for error in validation["errors"]:
                print(f"  ERROR: {error}")
            return None

        return tasks
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    print("LLM Data Loader Example:")
    print("=" * 70)

    # Create sample tasks
    from ..models.llm_task import create_prefill_task, create_decode_task

    tasks = [
        create_prefill_task("T1", "Qwen3", 30, 1.5, 0.0, 2048, 2),
        create_decode_task("T2", "Qwen3", 30, 1.0, 0.5, 512, 2),
        create_prefill_task("T3", "DeepSeek-R1", 671, 2.0, 1.0, 1024, 8),
    ]

    # Save to CSV
    output_path = "/tmp/test_llm_workload.csv"
    save_llm_tasks_to_csv(tasks, output_path)

    # Load back
    loaded_tasks = load_llm_tasks_from_csv(output_path)
    print(f"\nLoaded {len(loaded_tasks)} tasks")

    # Print summary
    print_dataset_summary(loaded_tasks)

    print("=" * 70)
