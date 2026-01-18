#!/usr/bin/env python3
"""
Generate workload where burst size exceeds cluster capacity.
"""

import pandas as pd
from pathlib import Path


def generate_overload_workload():
    """
    Generate workload where first burst exceeds cluster capacity.

    With 16 GPUs and Qwen3 tasks (TP=2):
    - Can only run 8 Qwen3 tasks at once
    - First burst has 15 tasks, so 7 will queue
    """
    tasks = []

    # First burst: 15 Qwen3 tasks, all arriving at t=0
    # With TP=2, need 2 GPUs each, so only 8 can run at once
    for i in range(15):
        tasks.append({
            "Task": f"T{i+1}",
            "Workload": "Qwen3",
            "Memory": 30,
            "Weight": 1 if i < 7 else 10,  # First 7 have low weight, last 8 have high weight
            "ArrivalTime": 0.0,
            "Phase": 1,
            "Tokens": 1000,
        })

    # Save to CSV
    output_path = "data/llm_workload/overload_test.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tasks).to_csv(output_path, index=False)

    print(f"Saved {len(tasks)} tasks to {output_path}")
    print("\nWorkload design:")
    print("- First burst: 15 Qwen3 tasks at t=0.0")
    print("- Each task needs 2 GPUs (TP=2)")
    print("- Cluster has 16 GPUs")
    print("- Only 8 tasks can run at once")
    print("- 7 tasks will wait in queue")
    print("\nWeight distribution:")
    print("- T1-T7: weight=1 (low priority)")
    print("- T8-T15: weight=10 (high priority)")
    print("\nExpected behavior:")
    print("- FIFO: Schedules T1-T8 first, then T9-T15")
    print("- WeightedFIFO: Schedules T8-T15 first (high weight), then T1-T7")


if __name__ == "__main__":
    generate_overload_workload()
