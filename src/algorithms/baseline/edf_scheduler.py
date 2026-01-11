"""
EDF 调度器：最早截止时间优先
"""

from typing import List

from ..base import BaseScheduler
from ...models.task import Task
from ...models.cluster import Cluster


class EDFScheduler(BaseScheduler):
    """
    EDF 调度器：最早截止时间优先

    策略：
    1. 按截止时间排序任务
    2. 依次调度到最早可用的 GPU
    """

    def __init__(self, cluster: Cluster):
        super().__init__()
        self.cluster = cluster

    def schedule(self, tasks: List[Task]) -> List[Task]:
        """
        执行 EDF 调度

        Args:
            tasks: 待调度的任务列表

        Returns:
            调度后的任务列表
        """
        # 重置集群状态
        self.cluster.reset()

        # 按截止时间排序
        sorted_tasks = sorted(tasks, key=lambda t: (t.deadline, t.task_id))

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

        # 对每个 GPU，找到最早可用时间
        best_gpu = None
        earliest_start = float('inf')

        for gpu in feasible_gpus:
            # 不能早于任务到达时间
            start_time = max(task.arrival_time, 0.0)
            earliest = gpu.find_earliest_start_time(task, start_time)

            if earliest < earliest_start:
                earliest_start = earliest
                best_gpu = gpu

        # 分配任务
        if best_gpu is not None:
            best_gpu.add_task(task, earliest_start)
            return True

        return False
