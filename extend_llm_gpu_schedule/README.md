# LLM GPU Scheduling Simulator

A simplified discrete-time simulator for studying LLM inference scheduling algorithms on heterogeneous GPU clusters.

## Overview

This simulator implements and compares different scheduling algorithms for LLM inference workloads, considering:
- **Tensor Parallelism (TP)**: Multi-GPU execution for large models
- **Task Heterogeneity**: Varying token counts, weights, and arrival times
- **GPU Constraints**: Memory limits and compute capacity
- **Priority Scheduling**: Weighted and duration-based prioritization

## Key Features

### Implemented Algorithms

| Algorithm | Strategy | Use Case |
|-----------|----------|----------|
| **FIFO** | First-In-First-Out by arrival time | Baseline, fair ordering |
| **WeightedFIFO** | Priority by weight, then arrival time | High-priority requests |
| **SRPT** | Shortest Remaining Processing Time | Minimize average completion |
| **WSRPT** | Weighted Shortest Remaining Processing Time | Minimize weighted completion |

### LLM Models Supported

- Llama3-8B, Llama3-70B
- Qwen3
- DeepSeek-R1 (32B, 70B)
- Mixtral-8x7B

### GPU Clusters

| Configuration | GPUs | Model |
|---------------|------|-------|
| `h100_8gpu` | 8 | H100 80GB |
| `h100_16gpu` | 16 | H100 80GB |
| `h100_32gpu` | 32 | H100 80GB |
| `h100_64gpu` | 64 | H100 80GB |

## How to Run

### Quick Start

Run the discriminating workload test:

```bash
python extend_llm_gpu_schedule/scripts/test_simplified.py
```

**Output:**
```
================================================================================
Simplified LLM GPU Scheduling Simulator Test
================================================================================

Cluster: h100_16gpu (16 GPUs)
Tasks: 20
  - T_low_1 to T_low_8: weight=1, tokens=2000 (long)
  - T_high_1 to T_high_12: weight=10, tokens=500 (short)

================================================================================
Algorithm Comparison Results
================================================================================
Algorithm            Weighted Wait   Avg Wait        Completed
--------------------------------------------------------------------------------
FIFO                 336.400         2.402           20
WeightedFIFO         108.040         1.802           20
SRPT                 108.040         1.802           20
WSRPT                108.040         1.802           20

WeightedFIFO vs FIFO (weighted waiting time):
  FIFO: 336.400
  WeightedFIFO: 108.040
  Improvement: 67.9%

SRPT vs FIFO (average waiting time):
  FIFO: 2.402
  SRPT: 1.802
  Improvement: 25.0%
```

### Custom Experiment Options

```bash
python extend_llm_gpu_schedule/scripts/run_simplified_experiments.py \
    --workload discriminating \
    --cluster h100_16gpu \
    --dt 0.01 \
    --max-time 5.0 \
    --algorithms FIFO WeightedFIFO SRPT WSRPT \
    --output results/llm_experiments/my_experiment.json
```

**Arguments:**
| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--workload` | `discriminating`, `staggered`, `both` | `both` | Workload pattern |
| `--cluster` | `h100_8gpu`, `h100_16gpu`, `h100_32gpu`, `h100_64gpu` | `h100_16gpu` | Cluster size |
| `--dt` | float (seconds) | `0.01` | Time step granularity |
| `--max-time` | float (seconds) | `5.0` | Maximum simulation duration |
| `--algorithms` | Space-separated list | All 4 | Which schedulers to run |
| `--output` | JSON file path | - | Save results to file |

## Simplified Simulator Architecture

### Design Principles

The simplified discrete-time simulator uses a three-stage execution model at each time step:

```
┌─────────────────────────────────────────────────────────┐
│                    Time Step (dt = 0.01s)               │
├─────────────────────────────────────────────────────────┤
│ 1. Complete finished tasks → free GPUs                  │
│ 2. Add newly arrived tasks to queue                     │
│ 3. Scheduler selects from queue (MULTIPLE tasks!)       │
│    → scheduler.select_next(queue) → prioritization!     │
└─────────────────────────────────────────────────────────┘
```

**Key Insight:** Scheduling decisions occur when GPUs become available and multiple tasks are waiting in the queue. This is where algorithm differences become visible!

### Why Algorithm Differences Matter

The simplified simulator ensures `select_next(queue)` is called with **multiple waiting tasks**:

```
Time: 0.0s
→ 20 tasks arrive simultaneously
→ Only 8 can run (16 GPUs / 2 GPUs per task)
→ Scheduler chooses which 8 to start
  - FIFO: T_low_1-8 (low weight, long)
  - WeightedFIFO: T_high_1-8 (high weight, short)
  - SRPT: T_high_1-8 (short duration)
```

**Result:** 67.9% improvement in weighted waiting time for priority-aware scheduling!

### Simplified vs Token-Based Simulator

| Feature | Simplified Simulator | Token-Based Simulator |
|---------|---------------------|----------------------|
| Time Model | Discrete steps (dt) | Event-driven (token-level) |
| Scheduling | Queue-based decisions | Immediate on arrival |
| Algorithm Differences | **Visible** ✅ | Not visible ❌ |
| Complexity | Low | High |
| Use Case | Algorithm comparison | Detailed simulation |

## Performance Metrics

| Metric | Formula | Optimized By |
|--------|---------|--------------|
| Weighted Waiting Time | Σ(weight × waiting_time) | WeightedFIFO, WSRPT |
| Average Waiting Time | Σ(waiting_time) / n | SRPT |
| Makespan | max(completion_time) | - |
| Completion Rate | completed / total | - |

## Example: Discriminating Workload

```python
# 20 tasks arrive at t=0
# Cluster: 16 GPUs (8 concurrent tasks with TP=2)

tasks = [
    # First 8: LOW weight (1), LONG duration (2000 tokens)
    create_prefill_task(f"T_low_{i+1}", "Qwen3", 30, 1.0, 0.0, 2000, 2)
    for i in range(8)
]

tasks += [
    # Next 12: HIGH weight (10), SHORT duration (500 tokens)
    create_prefill_task(f"T_high_{i+1}", "Qwen3", 30, 10.0, 0.0, 500, 2)
    for i in range(12)
]
```

**Behavior:**
- **FIFO**: Starts T_low_1-8 → high-weight tasks wait 2 seconds
- **WeightedFIFO/SRPT**: Start T_high_1-8 → high-weight tasks finish first

## File Structure

```
extend_llm_gpu_schedule/
├── README.md                              # This file
├── algorithms/
│   ├── base_llm_scheduler.py             # Abstract base + select_next()
│   └── baseline/
│       ├── llm_fifo.py                   # FIFO scheduler
│       ├── llm_weighted_fifo.py          # Weighted FIFO
│       └── srpt.py                       # SRPT + WSRPT
├── simulation/
│   ├── simplified_llm_simulator.py       # Discrete-time simulator ⭐
│   ├── llm_simulator.py                  # Token-based simulator (TODO)
│   └── token_event.py                    # Event queue (TODO)
├── models/
│   ├── llm_task.py                       # LLM task model
│   ├── llm_cluster.py                    # GPU cluster management
│   └── gpu_group.py                      # Multi-GPU allocation
├── config/
│   ├── llm_model_specs.py                # Model specifications
│   └── gpu_specs.py                      # GPU specifications
└── scripts/
    ├── test_simplified.py                # Quick test ⭐
    └── run_simplified_experiments.py     # Full experiment runner
```

## Design Insights

### Why the Token-Based Simulator Failed

Original design issue:
```python
def on_task_arrival(task, current_time, simulator):
    self.pending_queue.append(task)
    self._process_queue(current_time, simulator)  # Only 1 task in queue!
```

When called immediately on arrival, the queue only contains the new task. Sorting has no effect because there's nothing to choose from.

### Simplified Simulator Solution

```python
def step(dt=0.01):
    # 1. Complete finished tasks → free GPUs
    self._complete_finished_tasks()

    # 2. Add newly arrived tasks
    while tasks_arriving:
        self.queue.append(task)

    # 3. Scheduler chooses from queue (MULTIPLE tasks waiting!)
    while has_free_gpus() and self.queue:
        task = self.scheduler.select_next(self.queue)  # PRIORITIZATION!
        self.schedule(task)
```

Decisions happen when GPUs become available, with multiple tasks in queue.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

---


<!-- 
## Token-Based Simulator: TODO List

The simplified simulator successfully demonstrates algorithm differences for prefill-phase scheduling. However, a complete LLM inference simulator requires token-level event processing. The following components are pending implementation:

### 1. Two-Phase Execution Model

**Status:** Event handlers defined in `llm_simulator.py` but not integrated

**Required:**
- [ ] Implement prefill-to-decode phase transitions
- [ ] Handle PREFILL_COMPLETE event → trigger DECODE_START
- [ ] Track phase-specific token counts (prefill_tokens vs decode_tokens)
- [ ] Update task state on phase transitions

**Challenge:** The simplified simulator only handles prefill phase. Need to extend to handle both phases with proper state transitions.

### 2. Continuous Batching for Decode Phase

**Status:** `ContinuousBatchingManager` class exists but not tested

**Required:**
- [ ] Implement batch formation logic (group decode requests on same GPUs)
- [ ] Token-level processing for batches
- [ ] Completion tracking per task in batch
- [ ] Batch efficiency calculations (throughput scaling factors)
- [ ] Handle task completion at different times within batch

**Key Insight:** Decode phase can process multiple tasks simultaneously with improved throughput due to batch efficiency.

### 3. Roofline Performance Model

**Status:** `RooflineCalculator` skeleton defined

**Required:**
- [ ] Populate performance database: (GPU, Model, Phase) → tokens/s
- [ ] Implement compute-bound vs memory-bound throughput calculations
- [ ] Add TFLOPS and bandwidth parameters for each GPU model
- [ ] Calculate arithmetic intensity for each model/phase combination
- [ ] Validate against real-world benchmarks

**Data Needed:**
- H100: 989 TFLOPS, 3350 GB/s bandwidth
- A100: 628 TFLOPS, 2039 GB/s bandwidth
- Model-specific FLOPs per token (prefill vs decode)

### 4. Scheduler Callback Integration

**Issue:** Current event-driven design calls `on_task_arrival()` immediately, resulting in queue size = 1

**Required:**
- [ ] Batch arrival events (process multiple arrivals before scheduling)
- [ ] Defer scheduling until multiple tasks are pending
- [ ] Implement scheduling triggers on GPU completion events
- [ ] Add scheduler hooks for batch decode scheduling
- [ ] Handle preemption at token boundaries

**Design Challenge:** How to ensure multiple tasks are in queue when scheduler is called, in an event-driven system?

### 5. KV Cache Memory Management

**Status:** Not implemented

**Required:**
- [ ] Track KV cache memory allocation per GPU
- [ ] Calculate cache size per task (num_tokens × 2 × hidden_dim × precision)
- [ ] Update memory usage during decode phase (cache grows with each token)
- [ ] Handle cache eviction when memory is full
- [ ] Consider prefix caching for shared prompts

**Complexity:** KV cache can be comparable to model weights for long sequences.

### 6. Workload Generation

**Status:** Only synthetic discriminating workload implemented

**Required:**
- [ ] Trace-based workload generator from real LLM request logs
- [ ] Burst pattern support (simulating flash crowds)
- [ ] Varied token count distributions (Pareto, log-normal)
- [ ] Multi-model workloads (mix of Qwen3, DeepSeek-R1, etc.)
- [ ] Time-varying arrival rates
- [ ] Weighted vs unweighted request mixes

### 7. Performance Metrics Extension

**Required:**
- [ ] Time-varying GPU utilization (not just average)
- [ ] Batch efficiency metrics (tokens/s per GPU vs single request)
- [ ] KV cache memory utilization tracking
- [ ] Phase-specific metrics (prefill vs decode times)
- [ ] Token-level throughput measurements

### 8. Testing and Validation

**Required:**
- [ ] Unit tests for each event handler
- [ ] Integration tests for full two-phase workflows
- [ ] Validate against real-world inference traces
- [ ] Stress tests with large batch sizes
- [ ] Comparison with vLLM/Orca scheduling decisions -->

