#!/usr/bin/env python3
"""
Generate workload that forces algorithm trade-offs.

Key design:
1. 30 tasks arrive in burst
2. Cluster has 16 GPUs (can run 8 Qwen3 tasks at once)
3. Simulation limited to 0.5 seconds
4. NOT all tasks will complete
5. Scheduler must choose WHICH tasks to prioritize
"""

import pandas as pd
from pathlib import Path


def generate_forced_choice_workload():
    """
    Generate workload where schedulers must make trade-offs.

    Design:
    - 30 tasks arrive at t=0
    - Cluster has 16 GPUs (8 concurrent Qwen3 tasks max)
    - Simulation limited to 0.5 seconds (max_time in simulator)
    - Each task takes different time to complete
    - Scheduler choice affects which tasks complete
    """
    tasks = []

    # First 15 tasks: LOW weight (1-3), mixed durations
    for i in range(15):
        weight = 1 if i < 5 else (2 if i < 10 else 3)
        tokens = 1000 if i % 2 == 0 else 2000
        tasks.append({
            "Task": f"T_low_{i+1}",
            "Workload": "Qwen3",
            "Memory": 30,
            "Weight": weight,
            "ArrivalTime": 0.01 + i * 0.0001,  # Staggered
            "Phase": 1,
            "Tokens": tokens,
        })

    # Next 15 tasks: HIGH weight (8-10), mixed durations
    for i in range(15):
        weight = 10 if i < 5 else (9 if i < 10 else 8)
        tokens = 1000 if i % 2 == 0 else 2000
        tasks.append({
            "Task": f"T_high_{i+1}",
            "Workload": "Qwen3",
            "Memory": 30,
            "Weight": weight,
            "ArrivalTime": 0.02 + i * 0.0001,  # Staggered
            "Phase": 1,
            "Tokens": tokens,
        })

    # Save to CSV
    output_path = "data/llm_workload/forced_choice.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(tasks).sort_values("ArrivalTime")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(tasks)} tasks to {output_path}")
    print("\nWorkload design:")
    print("- 30 tasks arrive in burst (within 0.04 seconds)")
    print("- Cluster has 16 GPUs (max 8 concurrent Qwen3 tasks)")
    print("- Tasks have different weights and durations")
    print("\nExpected behavior:")
    print("- FIFO: Processes in arrival order (may complete low-weight tasks first)")
    print("- WeightedFIFO: Prioritizes high weight tasks")
    print("- SRPT: Prioritizes short duration tasks")
    print("\nNote: Results depend on actual simulator max_time limit")


if __name__ == "__main__":
    generate_forced_choice_workload()
