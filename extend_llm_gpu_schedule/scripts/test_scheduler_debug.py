#!/usr/bin/env python3
"""
Simple test to verify schedulers make different decisions.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_task import create_prefill_task
from models.llm_cluster import create_cluster_from_cluster_config
from algorithms.baseline.llm_fifo import FIFOScheduler
from algorithms.baseline.llm_weighted_fifo import WeightedFIFOScheduler
from algorithms.baseline.srpt import SRPTScheduler


def test_scheduler_ordering():
    """Test that schedulers produce different orderings."""
    # Create a small cluster (limited GPUs to force queueing)
    cluster = create_cluster_from_cluster_config("h100_8gpu")

    # Create test tasks with different weights, sizes, and arrival times
    # All arrive at t=0.0 but the simulator processes them in order
    # Use slight arrival differences to ensure stable sorting
    tasks = [
        create_prefill_task("T1", "Qwen3", 30, weight=1, arrival_time=0.01, tokens=1000, tp_degree=2),
        create_prefill_task("T2", "Qwen3", 30, weight=10, arrival_time=0.02, tokens=1000, tp_degree=2),
        create_prefill_task("T3", "Qwen3", 30, weight=5, arrival_time=0.03, tokens=5000, tp_degree=2),
        create_prefill_task("T4", "Qwen3", 30, weight=8, arrival_time=0.04, tokens=2000, tp_degree=2),
        create_prefill_task("T5", "Qwen3", 30, weight=3, arrival_time=0.05, tokens=3000, tp_degree=2),
        create_prefill_task("T6", "Qwen3", 30, weight=7, arrival_time=0.06, tokens=1500, tp_degree=2),
        create_prefill_task("T7", "Qwen3", 30, weight=2, arrival_time=0.07, tokens=4000, tp_degree=2),
    ]

    print("Test Tasks (more than cluster can handle):")
    for t in tasks:
        print(f"  {t.task_id}: weight={t.weight}, tokens={t.tokens}, arrival={t.arrival_time}")

    # Test FIFO scheduler
    print("\n" + "="*60)
    print("FIFO Scheduler (order by arrival time):")
    print("="*60)
    fifo = FIFOScheduler(cluster)

    for task in tasks:
        result = fifo.on_task_arrival(task, 0.0, None)
        status = "SCHEDULED" if task.is_scheduled() else "PENDING"
        print(f"  {task.task_id}: {status}")

    print(f"\nRemaining in queue: {len(fifo.pending_queue)}")
    if fifo.pending_queue:
        print("Queue order (by arrival time):")
        for t in fifo.pending_queue:
            print(f"  {t.task_id}: arrival={t.arrival_time}")

    # Test WeightedFIFO scheduler
    print("\n" + "="*60)
    print("WeightedFIFO Scheduler (order by weight then arrival):")
    print("="*60)
    cluster.reset()
    wf = WeightedFIFOScheduler(cluster)

    for task in tasks:
        result = wf.on_task_arrival(task, 0.0, None)
        status = "SCHEDULED" if task.is_scheduled() else "PENDING"
        print(f"  {task.task_id}: {status}")

    print(f"\nRemaining in queue: {len(wf.pending_queue)}")
    if wf.pending_queue:
        print("Queue order (by -weight, arrival):")
        for t in wf.pending_queue:
            print(f"  {t.task_id}: weight={t.weight}, arrival={t.arrival_time}")

    # Test SRPT scheduler
    print("\n" + "="*60)
    print("SRPT Scheduler (order by estimated remaining time):")
    print("="*60)
    cluster.reset()
    srpt = SRPTScheduler(cluster)

    for task in tasks:
        result = srpt.on_task_arrival(task, 0.0, None)
        status = "SCHEDULED" if task.is_scheduled() else "PENDING"
        print(f"  {task.task_id}: {status}")

    print(f"\nRemaining in queue: {len(srpt.pending_queue)}")
    if srpt.pending_queue:
        print("Queue order (by remaining_time, arrival):")
        import copy
        queue_copy = copy.copy(srpt.pending_queue)
        while queue_copy:
            remaining, arrival, task = queue_copy[0]
            print(f"  {task.task_id}: remaining={remaining:.4f}, arrival={arrival}")
            queue_copy.pop(0)

    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("With 8 GPUs and 2 GPUs per Qwen3 task:")
    print("- Only 4 tasks can be scheduled initially")
    print("- 3 tasks remain pending in queue")
    print("\nExpected pending queue order:")
    print("- FIFO: T5, T6, T7 (last to arrive)")
    print("- WeightedFIFO: T7(w2), T1(w1), T5(w3) - by weight")
    print("- SRPT: T7, T3, T5 - by estimated duration")


if __name__ == "__main__":
    test_scheduler_ordering()
