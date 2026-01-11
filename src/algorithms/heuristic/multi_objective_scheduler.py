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

    参数:
        cluster: GPU 集群
        alpha: 时间紧急度权重 (默认 1.0)
        beta: 执行效率权重 (默认 1.0)
        gamma: 显存适配权重 (默认 0.5)
        delta: 资源利用率权重 (默认 0.3)
    """

    def __init__(
        self,
        cluster: Cluster,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        delta: float = 0.3,
    ):
        super().__init__()
        self.cluster = cluster
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.current_time = 0.0

    def schedule(self, tasks: List[Task]) -> List[Task]:
        """
        执行多目标优化调度

        调度流程：
        1. 按到达时间初始化任务队列
        2. 迭代调度：
           a. 收集当前可调度任务（已到达且未调度）
           b. 为每个可调度任务计算所有可行 GPU 的评分
           c. 选择评分最高的组合进行调度
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
                self.current_time = min(t.arrival_time for t in unscheduled_tasks)
                continue

            best_assignment = self._find_best_assignment(ready_tasks)

            if not best_assignment:
                break

            best_task, best_gpu, best_start_time = best_assignment
            best_gpu.add_task(best_task, best_start_time)
            unscheduled_tasks.remove(best_task)

        self.scheduled_tasks = [t for t in tasks if t.is_scheduled()]
        return self.scheduled_tasks

    def _calculate_score(self, task: Task, gpu, start_time: float) -> float:
        """
        计算综合评分

        评分 = α * 紧急度 + β * 效率 + γ * 显存适配 + δ * 利用率

        Args:
            task: 任务对象
            gpu: GPU 对象
            start_time: 计划开始时间

        Returns:
            综合评分
        """
        execution_time = task.get_execution_time(gpu)
        completion_time = start_time + execution_time
        remaining_time = task.deadline - completion_time

        urgency = 10.0 if remaining_time <= 0 else 1.0 / max(remaining_time, 0.1)

        efficiency = gpu.scaling_factor / execution_time

        optimal_memory = gpu.memory_capacity / 2
        memory_diff = abs(task.memory - optimal_memory)
        memory_fit = 1.0 - (memory_diff / optimal_memory)

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

        Args:
            ready_tasks: 当前可调度的任务列表

        Returns:
            (best_task, best_gpu, best_start_time) 元组，如果没有可行分配则返回 None
        """
        best_task = None
        best_gpu = None
        best_score = float('-inf')
        best_start_time = float('inf')

        for task in ready_tasks:
            feasible_gpus = self.cluster.get_available_gpus(task)
            if not feasible_gpus:
                continue

            for gpu in feasible_gpus:
                start_time = max(task.arrival_time, self.current_time)
                earliest_start = gpu.find_earliest_start_time(task, start_time)

                if earliest_start == float('inf'):
                    continue

                score = self._calculate_score(task, gpu, earliest_start)

                if score > best_score:
                    best_score = score
                    best_task = task
                    best_gpu = gpu
                    best_start_time = earliest_start

        if best_task and best_gpu:
            return (best_task, best_gpu, best_start_time)
        return None


# 复杂度分析：
# 时间复杂度：O(n² × m)
#   - 外层循环：O(n) 次任务调度
#   - 内层循环：O(m) 个 GPU 评分计算
#   - find_earliest_start_time：O(n)
# 空间复杂度：O(n + m)
