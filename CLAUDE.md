# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Heterogeneous GPU Cluster Scheduling Simulator** - a Python research tool for studying and comparing scheduling algorithms in GPU clusters with heterogeneous computing capabilities. The simulator implements an NP-Hard scheduling problem with heterogeneous GPU speeds, memory constraints, time-sensitive deadlines, and dynamic task arrivals.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate task datasets with different load levels
python generate_tasks.py --task-id 3                      # Medium load (default)
python generate_tasks.py --task-id 4 --load-level high    # High load
python generate_tasks.py --task-id 5 --load-level extreme # Extreme load
python generate_tasks.py --task-ids 3,4,5                 # Generate multiple
python generate_tasks.py --all --load-level high          # Generate all missing

# Run experiments
python experiments/run_comparison.py --dataset data/tasks1.csv --cluster small
python experiments/run_comparison.py --full              # Run all dataset/cluster combinations

# Clean results
make clean
```

## Architecture Overview

### Core Models (`src/models/`)
- **Task**: Represents computational tasks with workload, memory requirements, deadlines, weights, and arrival times
- **GPU**: Models heterogeneous GPU resources with different scaling factors and memory capacities
- **Cluster**: Manages GPU collections with factory methods (`create_small_cluster()`, `create_medium_cluster()`, `create_large_cluster()`)

### Scheduling Algorithms (`src/algorithms/`)
- **BaseScheduler**: Abstract base class defining the unified interface that all schedulers must implement
- **Baseline algorithms**: FIFO, Greedy, SAGreedy (simulated annealing-based greedy)

New scheduling algorithms must extend `BaseScheduler` and implement the `schedule(tasks: List[Task]) -> List[Task]` method, returning tasks with `assigned_gpu`, `start_time`, and `completion_time` set.

### Simulation Engine (`src/simulation/`)
- **Simulator**: Event-driven simulation engine that manages task lifecycle through an event queue
- **Event/EventQueue**: Handles task arrivals and completions with timestamp-based processing
- **SimulationResult**: Captures outcomes with makespan and weighted tardiness metrics

### Metrics (`src/metrics/`)
- **MetricsCalculator**: Computes comprehensive performance metrics including weighted/average completion times, deadline miss counts/rates, weighted tardiness, GPU utilization (time and memory)

## GPU Configuration System

GPU configurations are defined in `config/gpu_configs.py` based on real NVIDIA specifications:

| Model | Memory | Scaling Factor | Base Performance |
|-------|--------|----------------|------------------|
| A100  | 80 GB  | 57.0x          | 19.5 TFLOPS      |
| A30   | 24 GB  | 30.0x (baseline) | 10.3 TFLOPS   |
| L40   | 48 GB  | 55.5x          | 19.1 TFLOPS      |

**Key design principle**: The scaling factors are designed so single GPUs cannot complete all tasks independently, forcing multi-GPU collaboration. The `base_scale` parameter controls difficulty - lower values make scheduling harder.

### Cluster Sizes
- **small**: 3 GPUs (1 of each model)
- **medium**: 6 GPUs (2 of each model)
- **large**: 9 GPUs (3 of each model)

## Load Levels for Dataset Generation

| Level    | Tasks | Workload | Arrival Range | Deadline Factor | Use Case |
|----------|-------|----------|---------------|-----------------|----------|
| low      | 500   | 50-400   | 0-5000        | 3.0-5.0x        | Basic functionality testing |
| medium   | 1000  | 50-800   | 0-2500        | 1.5-3.0x        | Default test scenario |
| high     | 1500  | 100-800  | 0-1500        | 0.8-1.5x        | Requires multi-GPU coordination |
| extreme  | 2000  | 200-800  | 0-1000        | 0.5-1.0x        | Stress test, requires parallelism |

Deadlines are calculated as: `deadline = arrival_time + (workload / baseline_scaling_factor) * deadline_factor`

## Important Implementation Notes

1. **State Management**: Always reset cluster and scheduler states between experiments using `reset()` methods
2. **Deep Copying**: Use `copy.deepcopy()` for tasks when running multiple algorithms to avoid state contamination
3. **Event-Driven Processing**: The simulation uses an event queue for accurate time management - tasks are processed based on arrival events
4. **Memory Constraints**: GPU memory capacity is the primary bottleneck for parallelism; tasks cannot exceed GPU memory limits
5. **Metric Calculation**: Some GPU utilization metrics require timeline analysis from GPU objects' internal state

## Results Organization

Experimental results are stored in `results/`:
- `metrics/`: CSV and JSON performance data
- `figures/`: Visualization charts (comparisons, Gantt charts, utilization)
- `schedules/`: Detailed task scheduling information
- `logs/`: Experiment execution logs

## Dataset Format

Tasks are stored in CSV format with columns:
- `Task`: Unique identifier
- `Workload`: Computational units
- `Memory`: GPU memory requirement (GB)
- `Deadline`: Absolute deadline time
- `Weight`: Priority/importance factor
- `ArrivalTime`: When task becomes available

## Adding New Scheduling Algorithms

1. Create a new file in `src/algorithms/baseline/` or `src/algorithms/[category]/`
2. Extend `BaseScheduler` from `src.algorithms.base`
3. Implement the `schedule()` method to return tasks with scheduling information populated
4. Add the algorithm to experiments in `experiments/run_comparison.py`
