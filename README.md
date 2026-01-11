# 异构 GPU 集群调度模拟器

一个用于研究异构 GPU 集群调度问题的 Python 仿真工具，支持多种调度算法和性能评估指标。

## 问题背景

异构 GPU 调度是一个经典的 NP-Hard 问题，需要同时考虑：
- **计算能力差异**：不同 GPU 的计算速度不同
- **显存约束**：任务对显存的需求不同，并发受显存容量限制
- **截止时间和优先级**：任务有不同的重要性和时间约束

## 特性

- 支持多种调度算法（FIFO、SPT、EDF、多目标优化）
- 事件驱动的仿真引擎
- 完整的性能评估指标
- 可视化结果对比
- 可扩展的 GPU 集群配置

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

| 型号 | 显存 | Scaling Factor |
|------|------|----------------|
| A100 | 80 GB | 2.0 |
| A30  | 24 GB | 1.0 |
| L40  | 48 GB | 1.5 |

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
# 生成任务数据集
python generate_tasks.py --task-id 3              # 生成 tasks3.csv
python generate_tasks.py --task-ids 3,4,5         # 生成多个数据集
python generate_tasks.py --all                    # 生成所有缺失的数据集

# 运行单个实验
python experiments/run_comparison.py --dataset data/tasks1.csv --cluster small

# 运行完整实验矩阵（自动扫描 data/ 目录所有数据集）
python experiments/run_comparison.py --full

# 指定算法
python experiments/run_comparison.py --dataset data/tasks1.csv --algorithms FIFO MultiObjective
```

## 评估指标

| 指标 | 说明 |
|------|------|
| 加权完成时间 | Σ(weight × completion_time) |
| 平均完成时间 | Σ(completion_time) / n |
| Deadline miss 数量 | completion_time > deadline 的任务数 |
| Deadline miss 率 | deadline miss 数量 / 总任务数 |
| 加权拖期 | Σ(weight × max(0, completion_time - deadline)) |
| Makespan | 最大完成时间 |
| GPU 时间利用率 | GPU 忙碌时间 / 总时间（合并重叠时间段，范围 0-1） |
| 平均并发任务数 | 平均同时运行的任务数（可 > 1，表示 GPU 同时处理多个任务） |
| GPU 峰值显存利用率 | 仿真过程中峰值时刻的显存使用率（范围 0-1） |
| GPU 平均显存利用率 | 时间加权的平均显存使用率（范围 0-1） |

**注**：加权完成时间、平均完成时间、deadline miss 相关指标在不同算法间可能相同，这是因为所有算法调度了相同的任务集，差异主要体现在 makespan 和资源利用率上。

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
│   │   └── heuristic/             # 启发式算法
│   ├── simulation/                # 仿真引擎
│   ├── metrics/                   # 评估指标
│   ├── utils/                     # 工具函数
│   └── visualization/             # 可视化
├── config/                        # 配置文件
├── data/                          # 数据集目录
│   ├── tasks1.csv                 # 数据集 1
│   ├── tasks2.csv                 # 数据集 2
│   └── ...                        # 更多数据集
├── experiments/                   # 实验脚本
├── results/                       # 结果输出
│   ├── metrics/                   # 指标数据
│   └── figures/                   # 可视化图表
├── generate_tasks.py              # 数据集生成脚本
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
- `heterogeneous_gpu_scheduling_analysis.md`

## 许可

MIT License
