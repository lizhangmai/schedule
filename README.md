# 异构 GPU 集群调度模拟器

一个用于研究异构 GPU 集群调度问题的 Python 仿真工具，支持多种调度算法和性能评估指标。

## 问题背景

异构 GPU 调度是一个经典的 NP-Hard 问题，需要同时考虑：
- **计算能力差异**：不同 GPU 的计算速度不同（基于真实 GPU 规格）
- **显存约束**：任务对显存的需求不同，并发受显存容量限制
- **截止时间和优先级**：任务有不同的重要性和时间约束

## 特性

- 支持多种调度算法（FIFO、SPT、EDF、多目标优化）
- 事件驱动的仿真引擎
- 完整的性能评估指标
- 可视化结果对比
- 可配置的 GPU 集群（从 `config/gpu_configs.py` 读取）
- 支持多种负载级别数据集生成

## 安装

```bash
# 克隆项目
git clone https://github.com/lizhangmai/schedule

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```python
from src.models.cluster import create_small_cluster
from src.utils.data_loader import load_tasks_from_csv
from src.algorithms.baseline.fifo_scheduler import FIFOScheduler
from src.algorithms.heuristic.multi_objective_scheduler import MultiObjectiveScheduler
from src.metrics.calculator import MetricsCalculator

# 加载任务数据
tasks = load_tasks_from_csv('data/tasks1.csv')

# 创建 GPU 集群
cluster = create_small_cluster()

# 运行调度算法
scheduler = MultiObjectiveScheduler(cluster)
scheduled_tasks = scheduler.schedule(tasks)

# 计算评估指标
from src.simulation.simulator import SimulationResult
result = SimulationResult(
    tasks=scheduled_tasks,
    makespan=max(t.completion_time for t in scheduled_tasks if t.completion_time),
    total_weighted_tardiness=sum(t.get_weighted_tardiness() for t in scheduled_tasks),
)
metrics = MetricsCalculator.calculate(scheduled_tasks, cluster, result)

print(f"Makespan: {metrics.makespan:.2f}")
print(f"Deadline Miss Rate: {metrics.deadline_miss_rate:.2%}")
print(f"GPU Time Utilization: {metrics.gpu_time_utilization:.2%}")
print(f"GPU Average Memory Utilization: {metrics.gpu_average_memory_utilization:.2%}")
```

## GPU 配置

配置文件：`config/gpu_configs.py`

| 型号 | 显存 | Scaling Factor | 说明 |
|------|------|----------------|------|
| A100 | 80 GB | 57.0 | 基于 19.5 TFLOPS FP32 |
| A30  | 24 GB | 30.0 | 基准 GPU (10.3 TFLOPS FP32) |
| L40  | 48 GB | 55.5 | 基于 19.1 TFLOPS FP32 |

**Scaling Factor 设计原则**：
- 基于 NVIDIA 官方 FP32 性能规格设置算力比值
- 单个 GPU 无法独立完成所有任务，必须多 GPU 协作
- 99.9% 的单个任务可在 deadline 内完成

## 调度算法

### 基线算法

- **FIFO**：按任务到达时间顺序调度
- **SPT**：最短处理时间优先
- **EDF**：最早截止时间优先

### 多目标优化调度器（核心算法）

综合评分函数：

```
Score = α·Urgency + β·Efficiency + γ·MemoryFit + δ·Utilization
```

| 维度 | 说明 |
|------|------|
| Urgency | 时间紧急度，deadline 越近分数越高 |
| Efficiency | 执行效率，GPU 越强、时间越短分数越高 |
| MemoryFit | 显存适配度，接近 GPU 容量一半时最优 |
| Utilization | 资源利用率，优先利用空闲 GPU |

**复杂度**：O(n² × m)

## 运行实验

```bash
# 生成任务数据集（支持不同负载级别）
python generate_tasks.py --task-id 3                      # 中等负载（默认）
python generate_tasks.py --task-id 4 --load-level high    # 高负载
python generate_tasks.py --task-id 5 --load-level extreme # 极高负载
python generate_tasks.py --task-ids 3,4,5                 # 生成多个
python generate_tasks.py --all --load-level high          # 生成所有缺失数据集

# 运行单个实验
python experiments/run_comparison.py --dataset data/tasks1.csv --cluster small

# 运行完整实验矩阵（自动扫描 data/ 目录所有数据集）
python experiments/run_comparison.py --full

# 指定算法
python experiments/run_comparison.py --dataset data/tasks1.csv --algorithms FIFO MultiObjective
```

### 负载级别说明

| 级别 | 任务数 | Workload | 到达时间范围 | Deadline因子 | 适用场景 |
|------|--------|----------|--------------|--------------|----------|
| low | 500 | 50-400 | 0-5000 | 3.0-5.0x | 轻松完成，测试基本功能 |
| medium | 1000 | 50-800 | 0-2500 | 1.5-3.0x | 默认测试场景 |
| high | 1500 | 100-800 | 0-1500 | 0.8-1.5x | 需要多 GPU 协作 |
| extreme | 2000 | 200-800 | 0-1000 | 0.5-1.0x | 压力测试，必须多 GPU 并行 |

## 评估指标

### Algorithm Comparison 图表（6个子图）

| 位置 | 指标 | 说明 |
|------|------|------|
| 左上 | Weighted Completion Time | 加权完成时间 Σ(weight × completion_time) |
| 中上 | Average Completion Time | 平均完成时间 |
| 右上 | Deadline Miss Count | 错过 deadline 的任务数 |
| 左下 | Deadline Miss Rate | Deadline 错过率 |
| 中下 | Weighted Tardiness | 加权拖期 Σ(weight × max(0, completion - deadline)) |
| 右下 | Makespan | 最大完成时间（总调度时长） |

### GPU Utilization 图表

| 指标 | 说明 |
|------|------|
| Time Utilization | GPU 时间利用率（忙碌时间/总时间，合并重叠段） |
| Average Memory Utilization | 时间加权的平均显存使用率 |

### 完整指标列表

| 指标 | 说明 | 范围 |
|------|------|------|
| 加权完成时间 | Σ(weight × completion_time) | - |
| 平均完成时间 | Σ(completion_time) / n | - |
| Deadline miss 数量 | completion_time > deadline 的任务数 | - |
| Deadline miss 率 | deadline miss 数量 / 总任务数 | 0-1 |
| 加权拖期 | Σ(weight × max(0, completion_time - deadline)) | - |
| Makespan | 最大完成时间 | - |
| GPU 时间利用率 | GPU 忙碌时间 / 总时间（合并重叠） | 0-1 |
| 平均并发任务数 | 平均同时运行的任务数（可 > 1） | ≥0 |
| 峰值显存利用率 | 仿真过程中峰值时刻的显存使用率 | 0-1 |
| 平均显存利用率 | 时间加权的平均显存使用率 | 0-1 |

## 项目结构

```
schedule/
├── src/
│   ├── models/                    # 数据模型
│   │   ├── task.py                # Task 类
│   │   ├── gpu.py                 # GPU 类
│   │   └── cluster.py             # Cluster 类
│   ├── algorithms/
│   │   ├── base.py                # 调度器基类
│   │   ├── baseline/              # 基线算法
│   │   │   ├── fifo_scheduler.py
│   │   │   ├── spt_scheduler.py
│   │   │   └── edf_scheduler.py
│   │   └── heuristic/             # 启发式算法
│   │       └── multi_objective_scheduler.py
│   ├── simulation/                # 仿真引擎
│   │   └── simulator.py
│   ├── metrics/                   # 评估指标
│   │   └── calculator.py
│   ├── utils/                     # 工具函数
│   │   └── data_loader.py
│   └── visualization/             # 可视化
│       └── plots.py
├── config/                        # 配置文件
│   └── gpu_configs.py             # GPU 配置
├── data/                          # 数据集目录
│   ├── tasks1.csv                 # 中等负载
│   ├── tasks2.csv                 # 中等负载
│   ├── tasks3.csv                 # 高负载
│   ├── tasks4.csv                 # 极高负载
│   └── ...                        # 更多数据集
├── experiments/                   # 实验脚本
│   └── run_comparison.py          # 算法对比实验
├── results/                       # 结果输出
│   ├── metrics/                   # CSV 和 JSON 指标数据
│   └── figures/                   # 可视化图表
├── generate_tasks.py              # 数据集生成脚本（支持负载级别）
└── requirements.txt
```

## 依赖

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## 参考文献

详细的问题形式化和复杂性分析请参考：
- `task.md`

## 许可

MIT License
