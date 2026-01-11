"""
多目标优化调度器

核心思想：综合考虑多个目标维度，为每个 (GPU, Task) 组合计算综合评分

评分维度：
    1. 时间紧急度 (Urgency): 考虑截止时间的紧迫程度
    2. 执行效率 (Efficiency): 考虑 GPU 计算能力与任务匹配度
    3. 显存适配度 (Memory Fit): 任务显存占用与 GPU 容量的匹配
    4. 资源利用率 (Utilization): 考虑 GPU 当前利用情况
"""

from typing import List, Tuple

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
        # 重置集群状态
        self.cluster.reset()

        # 按到达时间排序，作为基础顺序
        unscheduled_tasks = sorted(tasks, key=lambda t: (t.arrival_time, t.task_id))

        # 迭代调度所有任务
        while unscheduled_tasks:
            # 找到当前可调度的任务（已到达）
            current_time = self._get_current_time()
            ready_tasks = [t for t in unscheduled_tasks if t.arrival_time <= current_time]

            if not ready_tasks:
                # 如果没有准备好的任务，推进到下一个任务的到达时间
                current_time = min(t.arrival_time for t in unscheduled_tasks)

            # 为每个准备好的任务找到最优 GPU
            best_task = None
            best_gpu = None
            best_score = float('-inf')
            best_start_time = float('inf')

            for task in ready_tasks:
                feasible_gpus = self.cluster.get_available_gpus(task)
                if not feasible_gpus:
                    continue

                for gpu in feasible_gpus:
                    # 计算最早开始时间
                    start_time = max(task.arrival_time, current_time)
                    earliest_start = gpu.find_earliest_start_time(task, start_time)

                    if earliest_start == float('inf'):
                        continue

                    # 计算综合评分
                    score = self._calculate_score(task, gpu, earliest_start)

                    if score > best_score:
                        best_score = score
                        best_task = task
                        best_gpu = gpu
                        best_start_time = earliest_start

            # 调度选中的任务
            if best_task and best_gpu:
                best_gpu.add_task(best_task, best_start_time)
                unscheduled_tasks.remove(best_task)
            else:
                # 无法调度任何任务，跳出循环
                break

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

        # 1. 时间紧急度：距离截止时间越近，分数越高
        if remaining_time > 0:
            urgency = 1.0 / max(remaining_time, 0.1)
        else:
            # 已经超过截止时间，给予最高紧急度
            urgency = 10.0

        # 2. 执行效率：GPU 越强、执行时间越短，效率越高
        efficiency = gpu.scaling_factor / execution_time

        # 3. 显存适配度：任务显存接近 GPU 容量一半时最优
        #    这样可以保留空间用于并发执行
        optimal_memory = gpu.memory_capacity / 2
        memory_diff = abs(task.memory - optimal_memory)
        memory_fit = 1.0 - (memory_diff / optimal_memory)

        # 4. GPU 利用率：优先选择空闲 GPU（负载均衡）
        utilization = 1.0 - gpu.get_current_utilization(start_time)

        # 综合评分
        score = (
            self.alpha * urgency +
            self.beta * efficiency +
            self.gamma * memory_fit +
            self.delta * utilization
        )

        return score

    def _get_current_time(self) -> float:
        """
        获取当前仿真时间

        基于已调度任务的最早开始时间

        Returns:
            当前时间
        """
        if not self.cluster.gpus:
            return 0.0

        min_time = float('inf')
        for gpu in self.cluster.gpus:
            if gpu.timeline:
                # 获取最早的任务开始时间
                earliest = min(start for start, _, _ in gpu.timeline)
                min_time = min(min_time, earliest)

        return min_time if min_time != float('inf') else 0.0


# 复杂度分析：
# 时间复杂度：O(n² × m)
#   - 外层循环：O(n) 次任务调度
#   - 内层循环：O(m) 个 GPU 评分计算
#   - find_earliest_start_time：O(n)
# 空间复杂度：O(n + m)
