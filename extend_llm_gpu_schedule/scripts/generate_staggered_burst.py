#!/usr/bin/env python3
"""
Generate workload with controlled arrival order but burst timing.

Key design:
1. Tasks arrive within 0.1 seconds (burst)
2. But with controlled order (T1 first, T2 second, etc.)
3. Cluster can only handle 8 tasks at once
4. So later tasks must wait in queue
5. When GPUs free up, scheduler's queue ordering determines what runs next
"""

import pandas as pd
from pathlib import Path


def generate_staggered_burst_workload():
    """
    Generate workload where scheduler queue ordering matters.

    Design:
    - 12 tasks arrive within 0.1 seconds (burst)
    - Cluster has 16 GPUs, each task needs 2 GPUs (TP=2)
    - Only 8 can run at once, so 4 must wait
    - First 8 have LOW weight, LONG duration
    - Last 4 have HIGH weight, SHORT duration
    - After first 8 start completing, scheduler chooses from queue
    """
    tasks = []

    # First 8 tasks: LOW weight, LONG duration
    # These arrive first and will be running when later tasks arrive
    for i in range(8):
        tasks.append({
            "Task": f"T_low_long_{i+1}",
            "Workload": "Qwen3",
            "Memory": 30,
            "Weight": 1,  # Low weight
            "ArrivalTime": 0.01 + i * 0.001,  # Staggered arrival
            "Phase": 1,
            "Tokens": 5000,  # Long duration
        })

    # Next 4 tasks: HIGH weight, SHORT duration
    # These arrive slightly later and must wait
    for i in range(4):
        tasks.append({
            "Task": f"T_high_short_{i+1}",
            "Workload": "Qwen3",
            "Memory": 30,
            "Weight": 10,  # High weight
            "ArrivalTime": 0.02 + i * 0.001,  # Staggered arrival
            "Phase": 1,
            "Tokens": 500,  # Short duration
        })

    # Save to CSV (sorted by arrival time for readability)
    output_path = "data/llm_workload/staggered_burst.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(tasks).sort_values("ArrivalTime")
    df.to_csv(output_path, index=False)

    print(f"Saved {len(tasks)} tasks to {output_path}")
    print("\nWorkload design:")
    print("- 12 tasks arrive within 0.02 seconds (burst)")
    print("- Cluster has 16 GPUs, each task needs 2 GPUs (TP=2)")
    print("- Only 8 can run at once")
    print("- 4 tasks MUST wait in queue")
    print("\nTask groups:")
    print("- T_low_long_1 to T_low_long_8: weight=1, tokens=5000, arrival 0.01-0.017")
    print("- T_high_short_1 to T_high_short_4: weight=10, tokens=500, arrival 0.02-0.023")
    print("\nTimeline:")
    print("- t=0.01-0.017: T_low_long_1-8 arrive and start running (using all 16 GPUs)")
    print("- t=0.02-0.023: T_high_short_1-4 arrive and go to queue (no GPUs available)")
    print("- After T_low_long tasks complete, scheduler chooses from queue")
    print("\nExpected behavior:")
    print("- FIFO: Processes queue in arrival order (all low weight, long duration run first)")
    print("- WeightedFIFO: Reorders queue by weight (high weight tasks run first)")
    print("\nExpected weighted_waiting_time:")
    print("- FIFO: HIGH (high weight tasks wait for long low-weight tasks)")
    print("- WeightedFIFO: LOW (high weight tasks run as soon as GPUs free up)")


if __name__ == "__main__":
    generate_staggered_burst_workload()
