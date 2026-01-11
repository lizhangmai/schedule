"""
Task 类：表示一个需要调度的计算任务
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .gpu import GPU


@dataclass
class Task:
    """
    任务类：表示一个需要调度的计算任务

    属性:
        task_id: str - 任务唯一标识符
        workload: float - 工作量
        memory: int - 显存需求 (GB)
        deadline: float - 截止时间
        weight: int - 权重/优先级
        arrival_time: float - 到达时间

    状态属性:
        assigned_gpu: Optional[GPU] - 分配的GPU
        start_time: Optional[float] - 开始时间
        completion_time: Optional[float] - 完成时间
    """

    task_id: str
    workload: float
    memory: int
    deadline: float
    weight: int
    arrival_time: float

    # 调度后的状态（初始为 None）
    assigned_gpu: Optional["GPU"] = field(default=None, repr=False)
    start_time: Optional[float] = field(default=None, repr=False)
    completion_time: Optional[float] = field(default=None, repr=False)

    def get_execution_time(self, gpu: "GPU") -> float:
        """
        计算在指定GPU上的执行时间

        Args:
            gpu: GPU对象

        Returns:
            执行时间 = workload / gpu.scaling_factor
        """
        return self.workload / gpu.scaling_factor

    def get_tardiness(self) -> float:
        """
        计算拖期时间

        Returns:
            max(0, completion_time - deadline)
            如果未完成则返回 0
        """
        if self.completion_time is None:
            return 0.0
        return max(0.0, self.completion_time - self.deadline)

    def get_weighted_tardiness(self) -> float:
        """
        计算加权拖期

        Returns:
            weight * max(0, completion_time - deadline)
        """
        return self.weight * self.get_tardiness()

    def is_deadline_missed(self) -> bool:
        """
        判断是否超过截止时间

        Returns:
            如果完成时间 > 截止时间返回 True，否则返回 False
            如果未完成则返回 False
        """
        if self.completion_time is None:
            return False
        return self.completion_time > self.deadline

    def is_scheduled(self) -> bool:
        """判断任务是否已被调度"""
        return self.assigned_gpu is not None and self.start_time is not None

    def __lt__(self, other: "Task") -> bool:
        """
        支持多种比较方式，用于不同调度算法的排序
        默认按任务ID排序
        """
        return self.task_id < other.task_id

    def __repr__(self) -> str:
        gpu_info = f"GPU={self.assigned_gpu.gpu_id}" if self.assigned_gpu else "GPU=None"
        time_info = f"[{self.start_time:.2f}->{self.completion_time:.2f}]" if self.start_time is not None else "[unscheduled]"
        return f"Task({self.task_id}, {gpu_info}, {time_info})"


def sort_by_arrival_time(tasks: list[Task]) -> list[Task]:
    """按到达时间排序任务"""
    return sorted(tasks, key=lambda t: t.arrival_time)


def sort_by_deadline(tasks: list[Task]) -> list[Task]:
    """按截止时间排序任务"""
    return sorted(tasks, key=lambda t: t.deadline)


def sort_by_workload(tasks: list[Task]) -> list[Task]:
    """按工作量排序任务"""
    return sorted(tasks, key=lambda t: t.workload)
