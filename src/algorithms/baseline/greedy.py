"""
贪心调度器：按最早完成时间选择 GPU
"""

from typing import List

from ..base import BaseScheduler
from ...models.task import Task
from ...models.cluster import Cluster


class GreedyScheduler(BaseScheduler):
    """
    贪心调度器：按最早完成时间选择 GPU

    策略：
    1. 按到达时间排序任务
    2. 对每个任务，选择最早完成时间的 GPU（考虑 GPU 速度）
    3. 分配任务并更新 GPU 状态

    与 FIFO 的区别：
    - FIFO: 选择最早开始时间的 GPU
    - Greedy: 选择最早完成时间的 GPU（考虑 GPU scaling_factor）
    """

    def __init__(self, cluster: Cluster):
        super().__init__()
        self.cluster = cluster

    def schedule(self, tasks: List[Task]) -> List[Task]:
        """
        执行贪心调度

        Args:
            tasks: 待调度的任务列表

        Returns:
            调度后的任务列表
        """
        # 重置集群状态
        self.cluster.reset()

        # 按到达时间排序
        sorted_tasks = sorted(tasks, key=lambda t: (t.arrival_time, t.task_id))

        for task in sorted_tasks:
            self._schedule_task(task)

        self.scheduled_tasks = [t for t in sorted_tasks if t.is_scheduled()]
        return self.scheduled_tasks

    def _schedule_task(self, task: Task) -> bool:
        """
        调度单个任务

        Args:
            task: 任务对象

        Returns:
            是否成功调度
        """
        # 找到能容纳该任务的 GPU
        feasible_gpus = self.cluster.get_available_gpus(task)

        if not feasible_gpus:
            return False

        # 选择最早完成时间的 GPU
        best_gpu = None
        earliest_completion = float('inf')
        best_start_time = 0.0

        for gpu in feasible_gpus:
            # 找到最早可用开始时间（不能早于任务到达时间）
            start_time = max(task.arrival_time, gpu.find_earliest_start_time(task, task.arrival_time))
            # 计算完成时间
            completion_time = gpu.get_completion_time(task, start_time)

            if completion_time < earliest_completion:
                earliest_completion = completion_time
                best_start_time = start_time
                best_gpu = gpu

        # 分配任务
        if best_gpu is not None:
            best_gpu.add_task(task, best_start_time)
            return True

        return False
