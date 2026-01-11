# 异构 GPU 调度问题：形式化描述与复杂性分析

基于数据集 `tasks1.csv` 及异构计算环境（A100/A30/L40 等）的调度挑战，以下是该问题的形式化数学建模与计算复杂性证明。

## 1. 形式化问题描述 (Formal Problem Description)

该问题属于**异构计算环境下的带约束动态调度问题**，可建模为混合整数规划 (MIP) 问题。

### A. 集合与索引 (Sets & Indices)
* **任务集 (Tasks)**: $\mathcal{T} = \{T_1, T_2, \dots, T_n\}$，其中 $n$ 为任务总数。
* **GPU 资源集 (GPUs)**: $\mathcal{G} = \{G_1, G_2, \dots, G_m\}$，其中 $m$ 为 GPU 数量。

### B. 参数 (Parameters)
**任务属性**（来自 `tasks1.csv`）：
* $W_i$: 任务 $T_i$ 的工作量 (Workload)。
* $M_i$: 任务 $T_i$ 的显存需求 (Memory)。
* $D_i$: 任务 $T_i$ 的绝对截止时间 (Deadline)。
* $P_i$: 任务 $T_i$ 的权重或优先级 (Weight)。
* $A_i$: 任务 $T_i$ 的到达时间 (Arrival Time)。

**资源属性**：
* $S_j$: GPU $G_j$ 的计算能力/速度因子 (Scaling Factor)。
* $C_j$: GPU $G_j$ 的显存容量 (Memory Capacity)。

**推导参数**：
* $E_{ij} = W_i / S_j$: 任务 $T_i$ 在 GPU $G_j$ 上的**预计执行时间**。

### C. 决策变量 (Decision Variables)
我们需要决策任务的“分配”与“调度时间”：
1.  **分配变量**: $x_{ij} \in \{0, 1\}$
    * $x_{ij} = 1$: 任务 $T_i$ 分配给 GPU $G_j$。
    * $x_{ij} = 0$: 否则。
2.  **开始时间**: $s_i \in \mathbb{R}_{\ge 0}$
    * $s_i$: 任务 $T_i$ 的开始执行时间。

### D. 约束条件 (Constraints)

1.  **任务分配唯一性 (Assignment)**:
    每个任务必须被分配且仅被分配到一个 GPU 上。
    $$
    \sum_{j=1}^{m} x_{ij} = 1, \quad \forall i \in \{1, \dots, n\}
    $$

2.  **到达时间约束 (Arrival Time)**:
    任务必须在到达后方可开始执行。
    $$
    s_i \ge A_i, \quad \forall i \in \{1, \dots, n\}
    $$

3.  **显存容量约束 (Memory Capacity)**:
    在任意时刻 $t$，GPU $G_j$ 上运行的任务总显存占用不得超过其容量 $C_j$。
    定义活跃任务集合 $Active(t, j) = \{i \mid x_{ij}=1 \text{ 且 } s_i \le t < s_i + E_{ij} \}$。
    $$
    \sum_{i \in Active(t, j)} M_i \le C_j, \quad \forall t, \forall j
    $$

4.  **非抢占与互斥约束 (Non-preemption & Mutex)**:
    若任务 $T_i, T_k$ 分配给同一 GPU $G_j$ 且显存不足以支持并行（$M_i + M_k > C_j$），则它们的时间区间不可重叠。
    若 $x_{ij}=1, x_{kj}=1$，则：
    $$
    s_i + E_{ij} \le s_k \quad \lor \quad s_k + E_{kj} \le s_i
    $$

### E. 目标函数 (Objective Function)

为了平衡任务紧急性与系统效率，建议采用**最小化加权拖期 (Weighted Tardiness)**。

定义拖期 $L_i = \max(0, (s_i + \sum_{j} x_{ij} E_{ij}) - D_i)$。
$$
\text{Minimize } Z = \sum_{i=1}^{n} P_i \cdot L_i
$$

---

## 2. 复杂性分析 (Complexity Analysis)

**定理**: 异构 GPU 调度问题（含显存与时间约束）属于 **NP-Hard**。

**证明思路**:
通过将已知的 NP-Hard 问题归约 (Reduction) 到本问题的特例来证明。

### 归约 1: 无关联并行机调度问题 (Unrelated Parallel Machine Scheduling)
* **特例假设**:
    * 忽略显存约束 ($M_i=0$)。
    * 所有任务 $A_i=0$。
    * 无截止时间，目标为最小化最大完工时间 ($C_{max}$)。
* **映射**:
    * 此时问题退化为：$n$ 个任务在 $m$ 个处理速度不同的机器上调度，处理时间 $p_{ij}$ 取决于机器。
    * 这对应经典的 **$R || C_{max}$** 问题。
* **结论**:
    * 已知 $R || C_{max}$ 是强 NP-Hard 问题 (Lenstra et al., 1990)。
    * 原问题包含此特例，故原问题至少是 NP-Hard。

### 归约 2: 装箱问题 (Bin Packing Problem)
* **特例假设**:
    * 单 GPU ($m=1$)。
    * 任务需在同一时间窗口并行执行（时间约束极紧）。
    * 仅考虑显存能否容纳所有任务。
* **映射**:
    * 问题转化为：是否存在一种方式将具有体积 $M_i$ 的物品放入容量 $C_j$ 的箱子中。
    * 这对应经典的 **Bin Packing** 或 **Knapsack** 类问题。
* **结论**:
    * Bin Packing 是 NP-Hard 问题。

### 总结
由于该问题同时包含**组合优化**（资源分配，类比装箱）和**时间调度**（类比并行机调度）的难点，其实际求解空间随任务数呈指数级增长。对于 $N=1000$ 的规模，求精确解通常不可行，建议采用启发式算法（如 HEFT, Min-Min）或强化学习方法。
