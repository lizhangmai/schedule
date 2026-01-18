#!/usr/bin/env python3
"""
Generate workload that discriminates between scheduling algorithms.

Key design:
1. Tasks with different token counts (different durations)
2. Tasks with different weights
3. Burst arrival where tasks MUST wait in queue
4. When GPUs free up, scheduler choice matters
"""

import pandas as pd
from pathlib import Path


def generate_discriminating_workload():
    """
    Generate workload where scheduler choice matters.

    Design:
    - 20 tasks arrive at t=0
    - Cluster has 16 GPUs
    - Each task needs 2 GPUs (TP=2)
    - Only 8 can run at once, 12 must wait
    - Tasks have different weights AND durations
    """
    tasks = []

    # First 8 tasks: low weight, SHORT duration
    # These SHOULD NOT be scheduled first by WeightedFIFO/SRPT
    for i in range(8):
        tasks.append({
            "Task": f"T_short_low_{i+1}",
            "Workload": "Qwen3",
            "Memory": 30,
            "Weight": 1,  # Low weight
            "ArrivalTime": 0.0,
            "Phase": 1,
            "Tokens": 500,  # Short duration
        })

    # Next 12 tasks: HIGH weight, LONG duration
    # These SHOULD be scheduled first by WeightedFIFO
    for i in range(12):
        tasks.append({
            "Task": f"T_long_high_{i+1}",
            "Workload": "Qwen3",
            "Memory": 30,
            "Weight": 10,  # High weight
            "ArrivalTime": 0.0,
            "Phase": 1,
            "Tokens": 5000,  # Long duration
        })

    # Save to CSV
    output_path = "data/llm_workload/discriminating_test.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tasks).to_csv(output_path, index=False)

    print(f"Saved {len(tasks)} tasks to {output_path}")
    print("\nWorkload design:")
    print("- 20 tasks arrive at t=0.0")
    print("- Cluster has 16 GPUs, each task needs 2 GPUs (TP=2)")
    print("- Only 8 tasks can run at once")
    print("- 12 tasks MUST wait in queue")
    print("\nTask groups:")
    print("- T_short_low_1 to T_short_low_8: weight=1, tokens=500 (short)")
    print("- T_long_high_1 to T_long_high_12: weight=10, tokens=5000 (long)")
    print("\nExpected behavior:")
    print("- FIFO: Schedules in arrival order (T_short_low_1-8, then T_long_high_1-12)")
    print("- WeightedFIFO: Schedules high weight first (T_long_high_1-8, then...)")
    print("- SRPT: Schedules short duration first (T_short_low_1-8, then...)")


if __name__ == "__main__":
    generate_discriminating_workload()
