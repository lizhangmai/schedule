"""
仿真引擎：管理调度仿真的整个生命周期
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ..models.cluster import Cluster
from ..models.task import Task
from ..algorithms.base import BaseScheduler
from .event import Event, EventQueue, EventType


@dataclass
class SimulationResult:
    """
    仿真结果

    属性:
        tasks: 调度后的任务列表
        makespan: 最大完成时间
        total_weighted_tardiness: 总加权拖期
        metadata: 其他元数据
    """
    tasks: List[Task]
    makespan: float = 0.0
    total_weighted_tardiness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Simulator:
    """
    仿真引擎：管理调度仿真的整个生命周期

    使用事件驱动机制，支持动态任务到达和完成
    """

    def __init__(self, scheduler: BaseScheduler, cluster: Cluster):
        """
        初始化仿真引擎

        Args:
            scheduler: 调度器实例
            cluster: GPU 集群
        """
        self.scheduler = scheduler
        self.cluster = cluster
        self.event_queue = EventQueue()
        self.current_time = 0.0
        self.pending_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []

    def reset(self) -> None:
        """重置仿真状态"""
        self.current_time = 0.0
        self.event_queue.clear()
        self.pending_tasks.clear()
        self.completed_tasks.clear()
        self.cluster.reset()
        self.scheduler.reset()

    def run(self, tasks: List[Task]) -> SimulationResult:
        """
        运行仿真

        Args:
            tasks: 待调度的任务列表

        Returns:
            仿真结果对象
        """
        self.reset()
        self.pending_tasks = tasks.copy()

        # 1. 初始化：将所有任务的到达事件加入队列
        for task in tasks:
            arrival_event = Event(
                timestamp=task.arrival_time,
                event_type=EventType.TASK_ARRIVAL,
                data=task
            )
            self.event_queue.push(arrival_event)

        # 2. 注册事件处理器
        self.event_queue.register_handler(EventType.TASK_ARRIVAL, self._handle_task_arrival)
        self.event_queue.register_handler(EventType.TASK_COMPLETION, self._handle_task_completion)

        # 3. 事件循环
        while not self.event_queue.is_empty():
            event = self.event_queue.pop()
            if event is None:
                break

            # 更新仿真时间
            self.current_time = event.timestamp

            # 处理事件
            self.event_queue.dispatch(event)

        # 4. 计算结果
        return self._compute_result()

    def _handle_task_arrival(self, event: Event) -> None:
        """
        处理任务到达事件

        Args:
            event: 到达事件
        """
        task = event.data
        # 调用调度器进行调度决策
        # 注意：这里简化处理，实际调度逻辑由具体调度器实现
        self.scheduler.schedule(self.pending_tasks)

    def _handle_task_completion(self, event: Event) -> None:
        """
        处理任务完成事件

        Args:
            event: 完成事件
        """
        task = event.data
        self.completed_tasks.append(task)

    def _compute_result(self) -> SimulationResult:
        """
        计算并返回仿真结果

        Returns:
            仿真结果对象
        """
        scheduled_tasks = [t for t in self.pending_tasks if t.is_scheduled()]

        makespan = 0.0
        total_weighted_tardiness = 0.0

        for task in scheduled_tasks:
            if task.completion_time is not None:
                makespan = max(makespan, task.completion_time)
                total_weighted_tardiness += task.get_weighted_tardiness()

        return SimulationResult(
            tasks=scheduled_tasks,
            makespan=makespan,
            total_weighted_tardiness=total_weighted_tardiness,
            metadata={
                "scheduler": self.scheduler.get_algorithm_name(),
                "cluster_size": self.cluster.get_gpu_count(),
                "total_tasks": len(scheduled_tasks),
                "completed_tasks": len(self.completed_tasks),
            }
        )
