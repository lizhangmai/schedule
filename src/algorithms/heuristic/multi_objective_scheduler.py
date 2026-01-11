"""
多目标优化调度器

核心思想：综合考虑多个目标维度，为每个 (GPU, Task) 组合计算综合评分

评分维度：
    1. 时间紧急度 (Urgency): 考虑截止时间的紧迫程度
    2. 执行效率 (Efficiency): 考虑 GPU 计算能力与任务匹配度
    3. 显存适配度 (Memory Fit): 任务显存占用与 GPU 容量的匹配
    4. 资源利用率 (Utilization): 考虑 GPU 当前利用情况
"""

from typing import List, Optional, Tuple

from ..base import BaseScheduler
from ...models.task import Task
from ...models.cluster import Cluster


class MultiObjectiveScheduler(BaseScheduler):
    """
    多目标优化调度器

    综合评分函数：
    Score(task, gpu, t) = α·Urgency + β·Efficiency + γ·MemoryFit + δ·Utilization

    策略：
    1. 优先选择能最早开始的任务-GPU组合（类似 FIFO）
    2. 当有多个组合具有相近的开始时间时，使用多目标评分进行选择
    3. 评分考虑：deadline紧迫度、GPU效率、显存匹配度、负载均衡

    参数:
        cluster: GPU 集群
        alpha: 时间紧急度权重 (默认 1.0)
        beta: 执行效率权重 (默认 0.3)
        gamma: 显存适配权重 (默认 0.2)
        delta: 资源利用率权重 (默认 0.5)
    """

    def __init__(
        self,
        cluster: Cluster,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.5,
    ):
        super().__init__()
        self.cluster = cluster
        self.alpha = alpha  # 紧急度权重
        self.beta = beta    # 效率权重
        self.gamma = gamma  # 显存适配权重
        self.delta = delta  # 利用率权重（负载均衡）
        self.current_time = 0.0

    def schedule(self, tasks: List[Task]) -> List[Task]:
        """
        执行多目标优化调度

        调度流程：
        1. 按到达时间初始化任务队列
        2. 迭代调度：
           a. 收集当前可调度任务（已到达且未调度）
           b. 为每个可调度任务找到所有可行 GPU 的最早开始时间
           c. 首先按最早开始时间排序，然后使用多目标评分选择最优组合
        3. 返回调度结果

        Args:
            tasks: 待调度的任务列表

        Returns:
            调度后的任务列表
        """
        self.cluster.reset()

        unscheduled_tasks = sorted(tasks, key=lambda t: (t.arrival_time, t.task_id))

        while unscheduled_tasks:
            ready_tasks = [t for t in unscheduled_tasks if t.arrival_time <= self.current_time]

            if not ready_tasks:
                # 跳转到下一个任务到达时间
                next_arrival = min(t.arrival_time for t in unscheduled_tasks)
                self.current_time = next_arrival
                continue

            best_assignment = self._find_best_assignment(ready_tasks)

            if not best_assignment:
                # 当前 ready_tasks 无法调度，检查是否应该等待 GPU 可用
                # 找到当前正在运行的 GPU 中最早结束的时间
                earliest_end = None
                for gpu in self.cluster.gpus:
                    if gpu.timeline:
                        for start, end, _ in gpu.timeline:
                            if end > self.current_time:
                                if earliest_end is None or end < earliest_end:
                                    earliest_end = end

                if earliest_end is not None and earliest_end < max(t.arrival_time for t in ready_tasks):
                    # GPU 会在新任务到达前释放，等待
                    self.current_time = earliest_end
                    continue
                else:
                    # 无法调度，推进到下一个未调度任务的到达时间
                    next_arrival = min(t.arrival_time for t in unscheduled_tasks if t.arrival_time > self.current_time)
                    if next_arrival > self.current_time:
                        self.current_time = next_arrival
                        continue
                    # 所有任务都已到达但仍无法调度，终止
                    break

            best_task, best_gpu, best_start_time = best_assignment
            best_gpu.add_task(best_task, best_start_time)
            unscheduled_tasks.remove(best_task)

        self.scheduled_tasks = [t for t in tasks if t.is_scheduled()]
        return self.scheduled_tasks

    def _calculate_score(self, task: Task, gpu, start_time: float, completion_time: float) -> float:
        """
        计算综合评分

        评分策略：
        1. 紧急度：基于 deadline 剩余时间，越紧迫分数越高
        2. 效率：GPU 计算能力，归一化
        3. 显存适配：任务显存与 GPU 容量的匹配度
        4. 利用率：优先使用当前空闲的 GPU（负载均衡）

        所有指标归一化到 [0, 1] 范围

        Args:
            task: 任务对象
            gpu: GPU 对象
            start_time: 计划开始时间
            completion_time: 计划完成时间

        Returns:
            综合评分（越高越好）
        """
        # 1. 紧急度：基于 deadline 剩余时间
        remaining_time = task.deadline - completion_time
        if remaining_time < 0:
            # 已超期，紧急度最高
            urgency = 1.0
        elif remaining_time < 100:
            # 剩余时间很少，紧急度高
            urgency = 1.0 - (remaining_time / 100.0) * 0.5
        else:
            # 剩余时间充足，紧急度较低
            urgency = max(0.2, 1.0 - remaining_time / 1000.0)

        # 2. 效率：GPU 计算能力因子，范围 [1.0, 2.0]，归一化到 [0, 1]
        efficiency = (gpu.scaling_factor - 1.0) / 1.0

        # 3. 显存适配：任务显存与 GPU 容量的匹配度
        # 理想情况：任务显存接近 GPU 容量的 50-80%
        memory_ratio = task.memory / gpu.memory_capacity
        if 0.5 <= memory_ratio <= 0.8:
            memory_fit = 1.0
        elif memory_ratio < 0.5:
            # 显存浪费，轻微惩罚
            memory_fit = 0.5 + memory_ratio
        else:
            # 显存紧张，惩罚
            memory_fit = max(0, 1.0 - (memory_ratio - 0.8) / 0.2)

        # 4. 利用率：优先使用当前空闲的 GPU（负载均衡）
        utilization = 1.0 - gpu.get_current_utilization(start_time)

        return (
            self.alpha * urgency +
            self.beta * efficiency +
            self.gamma * memory_fit +
            self.delta * utilization
        )

    def _find_best_assignment(self, ready_tasks: List[Task]) -> Optional[Tuple[Task, object, float]]:
        """
        为准备好的任务找到最优 (task, gpu, start_time) 组合

        策略：
        1. 首先按最早开始时间排序（优先尽早调度）
        2. 对于开始时间相近的组合（差异 < 阈值），使用多目标评分选择

        Args:
            ready_tasks: 当前可调度的任务列表

        Returns:
            (best_task, best_gpu, best_start_time) 元组，如果没有可行分配则返回 None
        """
        candidates = []

        for task in ready_tasks:
            feasible_gpus = self.cluster.get_available_gpus(task)
            if not feasible_gpus:
                continue

            for gpu in feasible_gpus:
                start_time = max(task.arrival_time, self.current_time)
                earliest_start = gpu.find_earliest_start_time(task, start_time)

                if earliest_start == float('inf'):
                    continue

                execution_time = task.get_execution_time(gpu)
                completion_time = earliest_start + execution_time
                score = self._calculate_score(task, gpu, earliest_start, completion_time)

                candidates.append({
                    'task': task,
                    'gpu': gpu,
                    'start_time': earliest_start,
                    'completion_time': completion_time,
                    'score': score,
                })

        if not candidates:
            return None

        # 按开始时间排序
        candidates.sort(key=lambda x: x['start_time'])

        # 获取最早开始时间
        earliest_time = candidates[0]['start_time']

        # 定义时间窗口（容差）
        time_window = 10.0  # 10个时间单位内的视为"相近"

        # 筛选出时间窗口内的候选者
        window_candidates = [
            c for c in candidates
            if c['start_time'] <= earliest_time + time_window
        ]

        # 在时间窗口内选择评分最高的
        best = max(window_candidates, key=lambda x: x['score'])

        return (best['task'], best['gpu'], best['start_time'])


# 复杂度分析：
# 时间复杂度：O(n² × m)
#   - 外层循环：O(n) 次任务调度
#   - 内层循环：O(m) 个 GPU 评分计算
#   - find_earliest_start_time：O(n)
# 空间复杂度：O(n + m)
