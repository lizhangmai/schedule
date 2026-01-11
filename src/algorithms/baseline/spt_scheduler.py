"""
SPT 调度器：最短处理时间优先
"""

from typing import List, Tuple

from ..base import BaseScheduler
from ...models.task import Task
from ...models.cluster import Cluster


class SPTScheduler(BaseScheduler):
    """
    SPT 调度器：最短处理时间优先

    策略：
    1. 计算每个任务的最短可能执行时间（在最优 GPU 上）
    2. 按最短执行时间排序任务
    3. 依次为每个任务动态选择最早可用的 GPU

    改进：不再预处理固定 GPU 分配，而是在调度时动态选择
    这样可以更好地利用多 GPU 集群，实现负载均衡
    """

    def __init__(self, cluster: Cluster):
        super().__init__()
        self.cluster = cluster

    def schedule(self, tasks: List[Task]) -> List[Task]:
        """
        执行 SPT 调度

        Args:
            tasks: 待调度的任务列表

        Returns:
            调度后的任务列表
        """
        # 重置集群状态
        self.cluster.reset()

        # 计算每个任务的最短执行时间，用于排序
        task_exec_times: List[Tuple[Task, float]] = []

        for task in tasks:
            feasible_gpus = self.cluster.get_available_gpus(task)
            if not feasible_gpus:
                continue

            # 找到该任务的最短执行时间（用于排序）
            min_exec_time = min(
                task.get_execution_time(gpu) for gpu in feasible_gpus
            )
            task_exec_times.append((task, min_exec_time))

        # 按执行时间排序（最短优先）
        task_exec_times.sort(key=lambda x: (x[1], x[0].task_id))

        # 依次调度，动态选择最优 GPU
        for task, _ in task_exec_times:
            self._schedule_task(task)

        self.scheduled_tasks = [t for t in tasks if t.is_scheduled()]
        return self.scheduled_tasks

    def _schedule_task(self, task: Task) -> bool:
        """
        调度单个任务到最优 GPU（动态选择）

        策略：选择能最早开始执行的 GPU

        Args:
            task: 任务对象

        Returns:
            是否成功调度
        """
        # 找到能容纳该任务的 GPU
        feasible_gpus = self.cluster.get_available_gpus(task)
        if not feasible_gpus:
            return False

        # 对每个 GPU，找到最早可用时间，选择最早能开始的 GPU
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
        if best_gpu is not None and earliest_start != float('inf'):
            best_gpu.add_task(task, earliest_start)
            return True

        return False
