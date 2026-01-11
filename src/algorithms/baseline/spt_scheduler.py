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
    1. 计算每个任务在每个 GPU 上的执行时间
    2. 按（任务, GPU）组合的最短执行时间排序
    3. 依次调度到最优 GPU

    注意：在异构环境中，处理时间取决于 GPU 选择
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

        # 计算每个任务在每个 GPU 上的执行时间，找到最短组合
        task_gpu_pairs: List[Tuple[Task, float, str]] = []

        for task in tasks:
            feasible_gpus = self.cluster.get_available_gpus(task)
            if not feasible_gpus:
                continue

            # 找到该任务的最短执行时间及其 GPU
            min_time = float('inf')
            best_gpu_id = None

            for gpu in feasible_gpus:
                exec_time = task.get_execution_time(gpu)
                if exec_time < min_time:
                    min_time = exec_time
                    best_gpu_id = gpu.gpu_id

            if best_gpu_id:
                task_gpu_pairs.append((task, min_time, best_gpu_id))

        # 按执行时间排序
        task_gpu_pairs.sort(key=lambda x: (x[1], x[0].task_id))

        # 依次调度
        for task, _, gpu_id in task_gpu_pairs:
            self._schedule_task(task, gpu_id)

        self.scheduled_tasks = [t for t in tasks if t.is_scheduled()]
        return self.scheduled_tasks

    def _schedule_task(self, task: Task, preferred_gpu_id: str) -> bool:
        """
        调度单个任务到指定 GPU

        Args:
            task: 任务对象
            preferred_gpu_id: 首选 GPU ID

        Returns:
            是否成功调度
        """
        gpu = self.cluster.get_gpu(preferred_gpu_id)
        if gpu is None:
            return False

        # 找到最早可用时间
        start_time = max(task.arrival_time, 0.0)
        earliest_start = gpu.find_earliest_start_time(task, start_time)

        if earliest_start != float('inf'):
            gpu.add_task(task, earliest_start)
            return True

        return False
