# 异构 GPU 集群调度模拟器

Python 仿真工具，用于研究异构 GPU 集群调度问题。支持多种调度算法、完整的性能评估和可视化分析。

## 问题背景

异构 GPU 调度是一个 NP-Hard 问题，需要考虑：
- **计算能力差异**：不同 GPU 计算速度不同（基于真实规格）
- **显存约束**：并发任务受 GPU 显存容量限制
- **截止时间和优先级**：任务具有不同的时间约束和重要性

## 核心特性

- **调度算法**：FIFO、Greedy (EFT)、SAGreedy（模拟退火+贪心）
- **仿真引擎**：事件驱动的任务生命周期管理
- **性能评估**：9 项指标（加权完成时间、拖期、GPU 利用率等）
- **可视化**：算法对比图、Gantt 图、利用率分析
- **数据集生成**：支持 4 种负载级别（low/medium/high/extreme）

## 安装

```bash
git clone https://github.com/lizhangmai/schedule
pip install -r requirements.txt
```

## 快速开始

```python
from src.models.cluster import create_small_cluster
from src.utils.data_loader import load_tasks_from_csv
from src.algorithms.baseline.fifo_scheduler import FIFOScheduler
from src.simulation.simulator import SimulationResult
from src.metrics.calculator import MetricsCalculator

# 加载任务并运行调度
tasks = load_tasks_from_csv('data/tasks1.csv')
cluster = create_small_cluster()
scheduler = FIFOScheduler(cluster)
scheduled_tasks = scheduler.schedule(tasks)

# 计算性能指标
result = SimulationResult(
    tasks=scheduled_tasks,
    makespan=max(t.completion_time for t in scheduled_tasks),
    total_weighted_tardiness=sum(t.get_weighted_tardiness() for t in scheduled_tasks),
)
metrics = MetricsCalculator.calculate(scheduled_tasks, cluster, result)

print(f"Makespan: {metrics.makespan:.2f}")
print(f"Deadline Miss Rate: {metrics.deadline_miss_rate:.2%}")
print(f"GPU Time Utilization: {metrics.gpu_time_utilization:.2%}")
```

## GPU 配置

配置文件：`config/gpu_configs.py`

| 型号  | 显存   | Scaling Factor | 基准性能        |
|-------|--------|----------------|-----------------|
| A100  | 80 GB  | 57.0×          | 19.5 TFLOPS     |
| A30   | 24 GB  | 30.0× (基准)   | 10.3 TFLOPS     |
| L40   | 48 GB  | 55.5×          | 19.1 TFLOPS     |

**设计原则**：
- Scaling factor 基于 NVIDIA FP32 性能规格
- 单个 GPU 无法独立完成所有任务（需多 GPU 协作）
- 99.9% 的单个任务可在 deadline 内完成

## 调度算法

所有算法继承 `BaseScheduler`，实现 `schedule(tasks: List[Task]) -> List[Task]` 方法。

### FIFO

按任务到达时间顺序调度，选择最早可用的 GPU。时间复杂度：O(n × m)

### Greedy (EFT)

按到达时间排序，为每个任务选择最早完成时间（EFT）的 GPU。考虑 GPU 计算能力差异。时间复杂度：O(n × m × k)

### SAGreedy

模拟退火元启发式算法，在任务排列空间中搜索最优解：

- **初始解**：Greedy 算法结果
- **邻域生成**：基于温度动态调整拖期任务优先级（高温探索，低温利用）
- **评估函数**：加权拖期总和
- **接受准则**：模拟退火概率接受
- **参数**：初始温度 1000，降温系数 0.99，终止温度 1，最大迭代 1000

时间复杂度：O(iterations × n × m × k)

**实现优化**：
- 轻量级评估：GPU 时间占用列表，避免 deepcopy
- 多样化邻域：4 种拖期任务排序策略
- 实时进度：显示温度、当前 cost、最优 cost

## 运行实验

```bash
# 生成数据集
python generate_tasks.py --task-id 3                      # 中等负载（默认）
python generate_tasks.py --task-id 4 --load-level high    # 高负载
python generate_tasks.py --task-id 5 --load-level extreme # 极高负载
python generate_tasks.py --task-ids 3,4,5                 # 生成多个
python generate_tasks.py --all --load-level high          # 生成所有缺失

# 运行实验
python experiments/run_comparison.py --dataset data/tasks1.csv --cluster small
python experiments/run_comparison.py --full                # 运行所有组合
```

### 负载级别

| 级别    | 任务数 | Workload | 到达范围 | Deadline 因子 | 场景                  |
|---------|--------|----------|----------|---------------|-----------------------|
| low     | 500    | 50-400   | 0-5000   | 3.0-5.0×      | 基本功能测试          |
| medium  | 1000   | 50-800   | 0-2500   | 1.5-3.0×      | 默认测试场景          |
| high    | 1500   | 100-800  | 0-1500   | 0.8-1.5×      | 需要多 GPU 协作       |
| extreme | 2000   | 200-800  | 0-1000   | 0.5-1.0×      | 压力测试，必须多 GPU 并行 |

## 评估指标

### 算法对比指标（6 项）

| 指标                      | 说明                                            |
|---------------------------|-------------------------------------------------|
| Weighted Completion Time  | Σ(weight × (completion_time - arrival_time))    |
| Average Completion Time   | Σ(completion_time) / n                          |
| Deadline Miss Count       | completion_time > deadline 的任务数             |
| Deadline Miss Rate        | deadline miss 数量 / 总任务数                   |
| Weighted Tardiness        | Σ(weight × max(0, completion_time - deadline))  |
| Makespan                  | 最大完成时间（总调度时长）                      |

### GPU 利用率指标（3 项）

| 指标                      | 说明                          |
|---------------------------|-------------------------------|
| GPU Time Utilization      | GPU 忙碌时间 / 总时间（合并重叠） |
| Peak Memory Utilization   | 峰值时刻的显存使用率          |
| Average Memory Utilization | 时间加权的平均显存使用率      |

## 项目结构

```
schedule/
├── src/
│   ├── models/              # Task, GPU, Cluster 数据模型
│   ├── algorithms/          # 调度算法（FIFO, Greedy, SAGreedy）
│   ├── simulation/          # 事件驱动仿真引擎
│   ├── metrics/             # 性能评估指标计算
│   ├── utils/               # 数据加载工具
│   └── visualization/       # 可视化图表生成
├── config/
│   └── gpu_configs.py       # GPU 配置（A100/A30/L40）
├── data/                    # 任务数据集（tasks*.csv）
├── experiments/
│   └── run_comparison.py    # 算法对比实验脚本
├── results/                 # 实验结果（metrics/figures/schedules）
├── latex_prj/               # IEEEtran 格式论文
│   ├── main.tex             # 主论文源文件（含附录）
│   └── build/               # 编译输出（main.pdf）
├── generate_tasks.py        # 数据集生成脚本
├── requirements.txt         # Python 依赖
├── CLAUDE.md                # Claude Code 项目指南
└── task.md                  # 问题形式化描述
```

## LaTeX 论文项目

`latex_prj/` 目录包含基于 IEEEtran 模板的研究论文，记录了异构 GPU 调度问题的形式化、算法设计和实验评估。

### 论文结构

1. **形式化问题描述**：集合、索引、参数定义，数学模型和性能指标
2. **复杂性分析**：NP-Hard 证明
3. **算法设计**：FIFO、Greedy (EFT)、SAGreedy
4. **实验评估**：数据集配置、结果对比、Gantt 图分析
5. **附录**：7 个额外数据集性能表，24 个 Gantt 图

### 文档统计

- **页数**：4 页
- **表格**：10 个（3 正文 + 7 附录）
- **图**：26 个（2 正文 + 24 附录）
- **质量**：所有图表引用正确，无未定义引用，无宽度溢出

### 编译方法

```bash
cd latex_prj

# 推荐：使用 latexmk
latexmk -pdf main.tex

# 手动编译
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# 清理
latexmk -c
```

### 结果图表引用

论文直接引用 `results/` 目录生成的图表：

```latex
% 引用算法对比表
\input{../results/sf40_small_tasks1/metrics/tasks1_small_comparison.csv}

% 引用 Gantt 图
\includegraphics[width=0.32\textwidth]{../results/sf40_small_tasks1/figures/tasks1_small_gantt_FIFO_first50.png}
```

## 添加新调度算法

1. 在 `src/algorithms/baseline/` 或 `src/algorithms/[category]/` 创建新文件
2. 继承 `BaseScheduler` 并实现 `schedule()` 方法
3. 返回的任务必须包含：`assigned_gpu`、`start_time`、`completion_time`
4. 在 `schedule()` 开始时调用 `self.cluster.reset()`
5. 在 `experiments/run_comparison.py` 中注册算法

**模板**：
```python
from typing import List
from ..base import BaseScheduler
from ...models.task import Task
from ...models.cluster import Cluster

class MyScheduler(BaseScheduler):
    def __init__(self, cluster: Cluster):
        super().__init__()
        self.cluster = cluster

    def schedule(self, tasks: List[Task]) -> List[Task]:
        self.cluster.reset()
        # 实现调度逻辑
        self.scheduled_tasks = [t for t in tasks if t.is_scheduled()]
        return self.scheduled_tasks
```

## 重要实现注意事项

- **状态管理**：每次实验前必须重置集群和调度器状态（使用 `reset()`）
- **深拷贝**：运行多个算法时使用 `copy.deepcopy(tasks)` 避免状态污染
- **事件驱动**：仿真引擎使用事件队列进行时间管理
- **显存约束**：GPU 显存容量是并行度的主要瓶颈
- **指标计算**：部分 GPU 利用率指标需要分析 GPU 对象内部状态时间线

## 依赖

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0
- seaborn >= 0.11.0

## 参考资料

详细的问题形式化和复杂性分析：`task.md`

## 许可证

MIT License
