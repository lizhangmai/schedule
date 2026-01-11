"""
评估指标计算器
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd

from ..models.task import Task
from ..models.cluster import Cluster
from ..simulation.simulator import SimulationResult


@dataclass
class Metrics:
    """
    评估指标数据类

    属性:
        weighted_completion_time: 加权完成时间
        average_completion_time: 平均完成时间
        deadline_miss_count: Deadline miss 数量
        deadline_miss_rate: Deadline miss 率
        weighted_tardiness: 加权拖期总和
        makespan: 最大完成时间
        gpu_time_utilization: GPU 时间利用率 (0-1)
        gpu_average_concurrent_tasks: 平均并发任务数 (可>1)
        gpu_peak_memory_utilization: 峰值显存利用率 (0-1)
        gpu_average_memory_utilization: 平均显存利用率 (0-1)
    """
    weighted_completion_time: float
    average_completion_time: float
    deadline_miss_count: int
    deadline_miss_rate: float
    weighted_tardiness: float
    makespan: float
    gpu_time_utilization: float
    gpu_average_concurrent_tasks: float
    gpu_peak_memory_utilization: float
    gpu_average_memory_utilization: float

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "weighted_completion_time": self.weighted_completion_time,
            "average_completion_time": self.average_completion_time,
            "deadline_miss_count": float(self.deadline_miss_count),
            "deadline_miss_rate": self.deadline_miss_rate,
            "weighted_tardiness": self.weighted_tardiness,
            "makespan": self.makespan,
            "gpu_time_utilization": self.gpu_time_utilization,
            "gpu_average_concurrent_tasks": self.gpu_average_concurrent_tasks,
            "gpu_peak_memory_utilization": self.gpu_peak_memory_utilization,
            "gpu_average_memory_utilization": self.gpu_average_memory_utilization,
        }


class MetricsCalculator:
    """
    评估指标计算器

    计算调度结果的各项性能指标
    """

    @staticmethod
    def calculate(tasks: List[Task], cluster: Cluster, result: Optional[SimulationResult] = None) -> Metrics:
        """
        计算所有指标

        Args:
            tasks: 调度后的任务列表
            cluster: GPU 集群
            result: 仿真结果（可选）

        Returns:
            Metrics 对象
        """
        scheduled_tasks = [t for t in tasks if t.is_scheduled()]

        if not scheduled_tasks:
            return Metrics(
                weighted_completion_time=0.0,
                average_completion_time=0.0,
                deadline_miss_count=0,
                deadline_miss_rate=0.0,
                weighted_tardiness=0.0,
                makespan=0.0,
                gpu_time_utilization=0.0,
                gpu_average_concurrent_tasks=0.0,
                gpu_peak_memory_utilization=0.0,
                gpu_average_memory_utilization=0.0,
            )

        # 1. 加权完成时间
        weighted_completion_time = MetricsCalculator.weighted_completion_time(scheduled_tasks)

        # 2. 平均完成时间
        average_completion_time = MetricsCalculator.average_completion_time(scheduled_tasks)

        # 3. Deadline miss 相关
        deadline_miss_count = MetricsCalculator.deadline_miss_count(scheduled_tasks)
        deadline_miss_rate = MetricsCalculator.deadline_miss_rate(scheduled_tasks)

        # 4. 加权拖期
        weighted_tardiness = MetricsCalculator.weighted_tardiness(scheduled_tasks)

        # 5. Makespan
        makespan = MetricsCalculator.makespan(scheduled_tasks) if result is None else result.makespan

        # 6. GPU 利用率指标
        gpu_stats = MetricsCalculator.gpu_utilization(cluster, makespan)

        return Metrics(
            weighted_completion_time=weighted_completion_time,
            average_completion_time=average_completion_time,
            deadline_miss_count=deadline_miss_count,
            deadline_miss_rate=deadline_miss_rate,
            weighted_tardiness=weighted_tardiness,
            makespan=makespan,
            gpu_time_utilization=gpu_stats["time_utilization"],
            gpu_average_concurrent_tasks=gpu_stats["average_concurrent_tasks"],
            gpu_peak_memory_utilization=gpu_stats["peak_memory_utilization"],
            gpu_average_memory_utilization=gpu_stats["average_memory_utilization"],
        )

    @staticmethod
    def weighted_completion_time(tasks: List[Task]) -> float:
        """
        加权完成时间

        Σ(weight × completion_time)
        """
        return sum(t.weight * t.completion_time for t in tasks if t.completion_time is not None)

    @staticmethod
    def average_completion_time(tasks: List[Task]) -> float:
        """
        平均完成时间
        """
        completed = [t for t in tasks if t.completion_time is not None]
        if not completed:
            return 0.0
        return sum(t.completion_time for t in completed) / len(completed)

    @staticmethod
    def deadline_miss_count(tasks: List[Task]) -> int:
        """
        Deadline miss 数量
        """
        return sum(1 for t in tasks if t.is_deadline_missed())

    @staticmethod
    def deadline_miss_rate(tasks: List[Task]) -> float:
        """
        Deadline miss 率
        """
        scheduled = [t for t in tasks if t.is_scheduled()]
        if not scheduled:
            return 0.0
        miss_count = MetricsCalculator.deadline_miss_count(scheduled)
        return miss_count / len(scheduled)

    @staticmethod
    def weighted_tardiness(tasks: List[Task]) -> float:
        """
        加权拖期总和

        Σ(weight × max(0, completion_time - deadline))
        """
        return sum(t.get_weighted_tardiness() for t in tasks)

    @staticmethod
    def makespan(tasks: List[Task]) -> float:
        """
        最大完成时间
        """
        completed = [t for t in tasks if t.completion_time is not None]
        if not completed:
            return 0.0
        return max(t.completion_time for t in completed)

    @staticmethod
    def gpu_utilization(cluster: Cluster, total_time: float) -> Dict[str, float]:
        """
        GPU 利用率统计

        Args:
            cluster: GPU 集群
            total_time: 总仿真时间

        Returns:
            包含所有 GPU 利用率指标的字典
        """
        if total_time == 0 or not cluster.gpus:
            return {
                "time_utilization": 0.0,
                "average_concurrent_tasks": 0.0,
                "peak_memory_utilization": 0.0,
                "average_memory_utilization": 0.0,
            }

        total_time_util = 0.0
        total_concurrent_tasks = 0.0
        total_peak_memory_util = 0.0
        total_avg_memory_util = 0.0

        for gpu in cluster.gpus:
            total_time_util += gpu.get_compute_utilization(total_time)
            total_concurrent_tasks += gpu.get_average_concurrent_tasks(total_time)
            total_peak_memory_util += gpu.get_peak_memory_utilization()
            total_avg_memory_util += gpu.get_average_memory_utilization(total_time)

        return {
            "time_utilization": total_time_util / len(cluster.gpus),
            "average_concurrent_tasks": total_concurrent_tasks / len(cluster.gpus),
            "peak_memory_utilization": total_peak_memory_util / len(cluster.gpus),
            "average_memory_utilization": total_avg_memory_util / len(cluster.gpus),
        }


class ResultComparator:
    """
    多算法结果对比工具
    """

    @staticmethod
    def compare_algorithms(results: Dict[str, SimulationResult], cluster: Cluster) -> pd.DataFrame:
        """
        生成对比表格

        Args:
            results: 算法名 -> 仿真结果的字典
            cluster: GPU 集群

        Returns:
            DataFrame，列为算法，行为指标
        """
        data = {}

        for algo_name, result in results.items():
            metrics = MetricsCalculator.calculate(result.tasks, cluster, result)
            data[algo_name] = metrics.to_dict()

        df = pd.DataFrame(data).T
        return df

    @staticmethod
    def find_best_algorithm(results: Dict[str, SimulationResult], cluster: Cluster, metric: str) -> tuple[str, float]:
        """
        根据指定指标找到最佳算法

        Args:
            results: 算法名 -> 仿真结果的字典
            cluster: GPU 集群
            metric: 指标名称

        Returns:
            (算法名, 指标值) 元组
        """
        best_algo = None
        best_value = None

        for algo_name, result in results.items():
            metrics = MetricsCalculator.calculate(result.tasks, cluster, result)
            value = metrics.to_dict().get(metric)

            if value is None:
                continue

            if best_value is None or value < best_value:
                # 对于大多数指标，越小越好
                best_value = value
                best_algo = algo_name

        return best_algo, best_value
