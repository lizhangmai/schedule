"""
GPU 类：表示一个 GPU 计算资源
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, List, Tuple

if TYPE_CHECKING:
    from .task import Task


@dataclass
class GPU:
    """
    GPU 类：表示一个 GPU 计算资源

    属性:
        gpu_id: str - GPU 唯一标识符
        model: str - GPU 型号 (A100/A30/L40)
        memory_capacity: int - 显存容量 (GB)
        scaling_factor: float - 计算能力因子

    状态属性:
        timeline: List[Tuple[float, float, Task]] - 任务执行时间线
            每个元组表示 (start_time, completion_time, task)
    """

    gpu_id: str
    model: str
    memory_capacity: int
    scaling_factor: float

    # 时间线：(开始时间, 完成时间, 任务)
    timeline: List[Tuple[float, float, "Task"]] = field(default_factory=list, repr=False)

    def can_accommodate(self, task: "Task") -> bool:
        """
        检查是否有足够显存容纳任务

        Args:
            task: 任务对象

        Returns:
            如果任务显存需求 <= GPU 显存容量返回 True
        """
        return task.memory <= self.memory_capacity

    def get_available_memory_at(self, time: float) -> int:
        """
        获取指定时刻的可用显存

        Args:
            time: 查询时刻

        Returns:
            可用显存 (GB)
        """
        used_memory = 0
        for start, end, task in self.timeline:
            if start <= time < end:
                used_memory += task.memory
        return self.memory_capacity - used_memory

    def can_start_at(self, task: "Task", start_time: float) -> bool:
        """
        检查任务在指定时间是否可以开始（考虑显存）

        Args:
            task: 任务对象
            start_time: 计划开始时间

        Returns:
            如果任务可以在 start_time 开始返回 True
        """
        if not self.can_accommodate(task):
            return False

        execution_time = task.get_execution_time(self)
        completion_time = start_time + execution_time

        time_points = {start_time, completion_time}

        for t_start, t_end, _ in self.timeline:
            if not (t_end <= start_time or t_start >= completion_time):
                time_points.add(max(start_time, t_start))
                time_points.add(min(completion_time, t_end))

        for t in sorted(time_points):
            used_memory = task.memory
            for t_start, t_end, existing_task in self.timeline:
                if t_start <= t < t_end:
                    used_memory += existing_task.memory
            if used_memory > self.memory_capacity:
                return False

        return True

    def find_earliest_start_time(self, task: "Task", after_time: float = 0.0) -> float:
        """
        找到任务最早的可行开始时间

        Args:
            task: 任务对象
            after_time: 最早开始时间下限

        Returns:
            最早的可行开始时间，如果不可行则返回 float('inf')
        """
        if not self.can_accommodate(task):
            return float('inf')

        # 收集所有候选开始时间点
        candidate_times = [after_time]
        for start, end, _ in self.timeline:
            if end > after_time:
                candidate_times.append(end)

        # 按时间排序
        candidate_times = sorted(set(candidate_times))

        # 尝试每个候选时间
        for candidate in candidate_times:
            if self.can_start_at(task, candidate):
                return candidate

        return float('inf')

    def add_task(self, task: "Task", start_time: float) -> None:
        """
        添加任务到 GPU，更新时间线

        Args:
            task: 任务对象
            start_time: 开始时间
        """
        execution_time = task.get_execution_time(self)
        completion_time = start_time + execution_time
        self.timeline.append((start_time, completion_time, task))

        task.assigned_gpu = self
        task.start_time = start_time
        task.completion_time = completion_time

    def get_completion_time(self, task: "Task", start_time: float) -> float:
        """
        计算任务完成时间

        Args:
            task: 任务对象
            start_time: 开始时间

        Returns:
            完成时间
        """
        return start_time + task.get_execution_time(self)

    def get_compute_utilization(self, total_time: float) -> float:
        """
        计算计算资源时间利用率（GPU 忙碌时间 / 总时间）

        合并重叠的时间段，计算实际占用的时间。

        Args:
            total_time: 总仿真时间

        Returns:
            时间利用率 (0-1)
        """
        if total_time == 0 or not self.timeline:
            return 0.0

        # 收集所有时间点并标记为开始或结束
        events = []
        for start, end, _ in self.timeline:
            events.append((start, 1))   # 开始事件
            events.append((end, -1))    # 结束事件

        # 按时间排序
        events.sort(key=lambda x: x[0])

        # 计算实际忙碌时间（合并重叠时间段）
        busy_time = 0.0
        concurrent = 0
        prev_time = None

        for time, delta in events:
            if prev_time is not None and concurrent > 0:
                busy_time += time - prev_time
            concurrent += delta
            prev_time = time

        return busy_time / total_time

    def get_average_concurrent_tasks(self, total_time: float) -> float:
        """
        计算平均并发任务数（表示 GPU 同时执行多个任务的程度）

        Args:
            total_time: 总仿真时间

        Returns:
            平均并发任务数（可以 > 1）
        """
        if total_time == 0 or not self.timeline:
            return 0.0

        # 收集所有时间点
        events = []
        for start, end, _ in self.timeline:
            events.append((start, 1))   # 开始事件
            events.append((end, -1))    # 结束事件

        events.sort(key=lambda x: x[0])

        # 计算时间加权的平均并发任务数
        total_task_time = 0.0
        concurrent = 0
        prev_time = None

        for time, delta in events:
            if prev_time is not None:
                duration = time - prev_time
                total_task_time += concurrent * duration
            concurrent += delta
            prev_time = time

        return total_task_time / total_time

    def get_peak_memory_utilization(self) -> float:
        """
        计算峰值显存利用率

        Returns:
            峰值显存利用率 (0-1)
        """
        if not self.timeline:
            return 0.0

        time_points = set()
        for start, end, _ in self.timeline:
            time_points.update([start, end])

        max_usage = 0
        for t in time_points:
            used_memory = sum(
                task.memory for start, end, task in self.timeline
                if start <= t < end
            )
            max_usage = max(max_usage, used_memory)

        return max_usage / self.memory_capacity

    def get_average_memory_utilization(self, total_time: float) -> float:
        """
        计算时间加权的平均显存利用率

        Args:
            total_time: 总仿真时间

        Returns:
            平均显存利用率 (0-1)
        """
        if total_time == 0 or not self.timeline:
            return 0.0

        # 收集所有带任务信息的时间点事件
        events = []
        for start, end, task in self.timeline:
            events.append((start, 1, task.memory))   # 开始事件，+显存
            events.append((end, -1, task.memory))    # 结束事件，-显存

        events.sort(key=lambda x: x[0])

        # 计算时间加权的平均显存使用
        total_memory_time = 0.0
        concurrent_memory = 0
        prev_time = None

        for time, delta, memory in events:
            if prev_time is not None:
                duration = time - prev_time
                total_memory_time += concurrent_memory * duration
            concurrent_memory += delta * memory
            prev_time = time

        return total_memory_time / (self.memory_capacity * total_time)

    def get_current_utilization(self, current_time: float) -> float:
        """
        获取当前时刻的 GPU 利用率（用于调度决策）

        Args:
            current_time: 当前时间

        Returns:
            当前利用率 (0-1)
        """
        used_memory = sum(
            task.memory for start, end, task in self.timeline
            if start <= current_time < end
        )
        return used_memory / self.memory_capacity

    def get_task_count(self) -> int:
        """获取时间线上的任务数量"""
        return len(self.timeline)

    def __repr__(self) -> str:
        return f"GPU({self.gpu_id}, {self.model}, {self.memory_capacity}GB, sf={self.scaling_factor})"
