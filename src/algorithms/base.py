"""
调度器基类：定义所有调度算法的统一接口
"""

from abc import ABC, abstractmethod
from typing import List

from ..models.task import Task


class BaseScheduler(ABC):
    """
    调度器基类：定义所有调度算法的统一接口
    """

    def __init__(self):
        self.scheduled_tasks: List[Task] = []

    @abstractmethod
    def schedule(self, tasks: List[Task]) -> List[Task]:
        """
        调度方法：输入任务列表，返回调度后的任务列表

        Args:
            tasks: 待调度的任务列表

        Returns:
            调度后的任务列表，每个任务包含：
            - assigned_gpu: 分配的 GPU
            - start_time: 开始时间
            - completion_time: 完成时间
        """
        pass

    def get_algorithm_name(self) -> str:
        """返回算法名称"""
        return self.__class__.__name__

    def reset(self) -> None:
        """重置调度器状态"""
        self.scheduled_tasks.clear()
