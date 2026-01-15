"""
结果可视化工具
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..metrics.calculator import Metrics, MetricsCalculator
from ..models.cluster import Cluster
from ..simulation.simulator import SimulationResult


class PlotGenerator:
    """
    结果可视化工具

    生成算法对比图表、GPU 利用率曲线等
    """

    # 设置风格
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10

    @staticmethod
    def plot_algorithm_comparison(
        results: Dict[str, SimulationResult],
        cluster: Cluster,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        生成算法对比柱状图

        Args:
            results: 算法名 -> 仿真结果的字典
            cluster: GPU 集群
            save_path: 保存路径
            show: 是否显示图表
        """
        # 计算每个算法的指标
        metrics_data = {}
        for algo_name, result in results.items():
            metrics = MetricsCalculator.calculate(result.tasks, cluster, result)
            metrics_data[algo_name] = metrics.to_dict()

        df = pd.DataFrame(metrics_data).T

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Algorithm Comparison", fontsize=16)

        # 要绘制的指标
        plot_metrics = [
            ("weighted_completion_time", "Weighted Completion Time"),
            ("average_completion_time", "Average Completion Time"),
            ("deadline_miss_count", "Deadline Miss Count"),
            ("deadline_miss_rate", "Deadline Miss Rate"),
            ("weighted_tardiness", "Weighted Tardiness"),
            ("makespan", "Makespan"),
        ]

        for ax, (metric, title) in zip(axes.flat, plot_metrics):
            if metric in df.columns:
                df[metric].plot(kind="bar", ax=ax)
                ax.set_title(title)
                ax.set_xlabel("Algorithm")
                ax.set_ylabel("Value")
                ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_gpu_utilization(
        results: Dict[str, SimulationResult],
        cluster: Cluster,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        绘制 GPU 利用率对比

        Args:
            results: 算法名 -> 仿真结果的字典
            cluster: GPU 集群
            save_path: 保存路径
            show: 是否显示图表
        """
        algorithms = []
        compute_utils = []
        memory_utils = []

        for algo_name, result in results.items():
            metrics = MetricsCalculator.calculate(result.tasks, cluster, result)
            algorithms.append(algo_name)
            compute_utils.append(metrics.gpu_time_utilization)
            memory_utils.append(metrics.gpu_average_memory_utilization)

        x = np.arange(len(algorithms))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width / 2, compute_utils, width, label="Compute Utilization", alpha=0.8)
        bars2 = ax.bar(x + width / 2, memory_utils, width, label="Memory Utilization", alpha=0.8)

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Utilization")
        ax.set_title("GPU Utilization Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend()
        ax.set_ylim([0, 1])

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_gantt_chart(
        result: SimulationResult,
        cluster: Cluster,
        save_path: Optional[str] = None,
        show: bool = True,
        max_tasks: Optional[int] = 50,
    ) -> None:
        """
        绘制调度甘特图

        Args:
            result: 仿真结果
            cluster: GPU 集群
            save_path: 保存路径
            show: 是否显示图表
            max_tasks: 最大显示任务数，None 表示显示全部任务
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # 按 GPU 分组任务
        gpu_tasks = {}
        for gpu in cluster.gpus:
            gpu_tasks[gpu.gpu_id] = []

        # max_tasks=None 时显示全部任务
        if max_tasks is None:
            scheduled_tasks = [t for t in result.tasks if t.is_scheduled()]
        else:
            scheduled_tasks = [t for t in result.tasks if t.is_scheduled()][:max_tasks]

        for task in scheduled_tasks:
            if task.assigned_gpu:
                gpu_id = task.assigned_gpu.gpu_id
                gpu_tasks[gpu_id].append(task)

        # 绘制甘特图
        y_pos = 0
        gpu_labels = []
        y_positions = {}

        for gpu_id, tasks in gpu_tasks.items():
            gpu_labels.append(gpu_id)
            y_positions[gpu_id] = y_pos

            for task in tasks:
                if task.start_time is not None and task.completion_time is not None:
                    duration = task.completion_time - task.start_time
                    color = "#ff6b6b" if task.is_deadline_missed() else "#51cf66"
                    ax.barh(
                        y_pos,
                        duration,
                        left=task.start_time,
                        height=0.8,
                        edgecolor="black",
                        linewidth=0.5,
                        color=color,
                        alpha=0.7,
                    )
                    # 仅在显示部分任务时添加任务标签（全部任务时太密集）
                    if max_tasks is not None:
                        ax.text(
                            task.start_time + duration / 2,
                            y_pos,
                            task.task_id,
                            ha="center",
                            va="center",
                            fontsize=7,
                        )
            y_pos += 1

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(gpu_labels)
        ax.set_xlabel("Time")
        ax.set_ylabel("GPU")
        # max_tasks=None 时显示 "all tasks"，否则显示数量
        task_count_str = "all" if max_tasks is None else str(min(max_tasks, len([t for t in result.tasks if t.is_scheduled()])))
        ax.set_title(f"Schedule Gantt Chart (showing {task_count_str} tasks)")

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#51cf66", label="On Time"),
            Patch(facecolor="#ff6b6b", label="Deadline Miss"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_task_timeline(
        result: SimulationResult,
        save_path: Optional[str] = None,
        show: bool = True,
        max_tasks: int = 100,
    ) -> None:
        """
        绘制任务完成时间分布

        Args:
            result: 仿真结果
            save_path: 保存路径
            show: 是否显示图表
            max_tasks: 最大显示任务数
        """
        scheduled_tasks = [t for t in result.tasks if t.is_scheduled()][:max_tasks]

        if not scheduled_tasks:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        task_ids = [t.task_id for t in scheduled_tasks]
        completion_times = [t.completion_time for t in scheduled_tasks]
        deadlines = [t.deadline for t in scheduled_tasks]
        colors = ["#ff6b6b" if t.is_deadline_missed() else "#51cf66" for t in scheduled_tasks]

        x = np.arange(len(task_ids))
        width = 0.35

        bars1 = ax.bar(x - width / 2, deadlines, width, label="Deadline", alpha=0.7, color="#4dabf7")
        bars2 = ax.bar(x + width / 2, completion_times, width, label="Completion Time", color=colors)

        ax.set_xlabel("Task ID")
        ax.set_ylabel("Time")
        ax.set_title("Task Deadline vs Completion Time")
        ax.set_xticks(x[::5])  # 每5个任务显示一个标签
        ax.set_xticklabels(task_ids[::5], rotation=45, ha="right")
        ax.legend()

        # 添加图例说明颜色
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#51cf66", label="On Time"),
            Patch(facecolor="#ff6b6b", label="Deadline Miss"),
        ]
        ax.legend(handles=legend_elements + [bars1, bars2], loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()
