"""
Visualization tools for LLM GPU scheduling experiments.

Generates algorithm comparison charts, GPU utilization plots,
token-level Gantt charts, and batch efficiency visualizations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from ..models.llm_task import Phase, LLMTask
from ..models.llm_cluster import LLMCluster
from ..models.gpu_group import GPUGroup
from ..metrics.llm_metrics import LLMMetrics


class LLMPlotGenerator:
    """
    Result visualization tool for LLM GPU scheduling.

    Generates algorithm comparison charts, GPU utilization curves,
    token-level Gantt charts, and batch efficiency plots.
    """

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 9

    # Color scheme
    PREFILL_COLOR = "#4dabf7"    # Blue for prefill (compute-bound)
    DECODE_COLOR = "#51cf66"     # Green for decode (memory-bound)
    WAIT_COLOR = "#ffd43b"       # Yellow for waiting
    OVERLAP_COLOR = "#ff8787"    # Red for overlapping/batching

    @staticmethod
    def plot_algorithm_comparison(
        results: Dict[str, Dict],
        cluster: LLMCluster,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Generate algorithm comparison bar charts.

        Args:
            results: Dictionary mapping algorithm name to metrics dict
            cluster: LLM cluster object
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        # Extract metrics data
        metrics_data = {}
        for algo_name, result in results.items():
            if result.get("success", False):
                metrics_data[algo_name] = result["metrics"]

        if not metrics_data:
            print("No successful results to plot")
            return

        df = pd.DataFrame(metrics_data).T

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("LLM Scheduling Algorithm Comparison", fontsize=16, fontweight="bold")

        # Metrics to plot
        plot_metrics = [
            ("weighted_waiting_time", "Weighted Waiting Time"),
            ("average_waiting_time", "Average Waiting Time"),
            ("throughput", "Throughput (tasks/sec)"),
            ("batch_efficiency", "Batch Efficiency"),
            ("gpu_utilization", "GPU Utilization"),
            ("makespan", "Makespan (sec)"),
        ]

        for ax, (metric, title) in zip(axes.flat, plot_metrics):
            if metric in df.columns:
                values = df[metric]
                bars = ax.bar(range(len(values)), values, color="#4dabf7", alpha=0.8, edgecolor="black", linewidth=0.5)
                ax.set_title(title, fontweight="bold")
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(df.index, rotation=45, ha="right")
                ax.set_ylabel("Value")

                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom" if height > 0 else "top",
                        fontsize=7,
                    )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved algorithm comparison plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_gpu_utilization(
        results: Dict[str, Dict],
        cluster: LLMCluster,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot GPU utilization comparison across algorithms.

        Args:
            results: Dictionary mapping algorithm name to metrics dict
            cluster: LLM cluster object
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        algorithms = []
        compute_utils = []
        memory_utils = []

        for algo_name, result in results.items():
            if result.get("success", False):
                metrics = result["metrics"]
                algorithms.append(algo_name)
                compute_utils.append(metrics.get("gpu_utilization", 0.0))
                memory_utils.append(metrics.get("memory_utilization", 0.0))

        if not algorithms:
            print("No successful results to plot")
            return

        x = np.arange(len(algorithms))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(
            x - width / 2,
            compute_utils,
            width,
            label="Compute Utilization",
            color="#4dabf7",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            memory_utils,
            width,
            label="Memory Utilization",
            color="#51cf66",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xlabel("Algorithm", fontweight="bold")
        ax.set_ylabel("Utilization", fontweight="bold")
        ax.set_title("GPU Utilization Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)

        # Add value labels
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
            print(f"  Saved GPU utilization plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_token_gantt_chart(
        tasks: List[LLMTask],
        cluster: LLMCluster,
        save_path: Optional[str] = None,
        show: bool = False,
        max_tasks: Optional[int] = 50,
    ) -> None:
        """
        Plot token-level Gantt chart showing prefill and decode phases.

        Args:
            tasks: List of LLM tasks with scheduling information
            cluster: LLM cluster object
            save_path: Path to save the figure
            show: Whether to display the plot
            max_tasks: Maximum number of tasks to display
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Filter scheduled tasks
        scheduled_tasks = [t for t in tasks if t.is_scheduled()]

        if max_tasks is not None:
            scheduled_tasks = scheduled_tasks[:max_tasks]

        if not scheduled_tasks:
            print("No scheduled tasks to plot")
            return

        # Group tasks by GPU group (create group ID from list of GPUs)
        gpu_group_tasks = {}
        for task in scheduled_tasks:
            if task.assigned_gpu_group:
                # Create a group identifier from the GPU list
                if isinstance(task.assigned_gpu_group, list):
                    # Sort GPU IDs to create consistent group identifier
                    gpu_ids = sorted([str(g) for g in task.assigned_gpu_group])
                    group_id = f"Group_{'_'.join(gpu_ids[:2])}"  # Use first 2 GPUs for ID
                else:
                    group_id = str(task.assigned_gpu_group)

                if group_id not in gpu_group_tasks:
                    gpu_group_tasks[group_id] = []
                gpu_group_tasks[group_id].append(task)

        # Sort groups by ID for consistent display
        sorted_groups = sorted(gpu_group_tasks.keys())

        # Plot tasks
        y_pos = 0
        y_positions = {}
        y_labels = []

        for group_id in sorted_groups:
            y_labels.append(group_id)
            y_positions[group_id] = y_pos

            for task in gpu_group_tasks[group_id]:
                # Plot prefill phase (start_time to prefill_completion_time)
                if task.start_time is not None and task.prefill_completion_time is not None:
                    prefill_duration = task.prefill_completion_time - task.start_time
                    ax.barh(
                        y_pos,
                        prefill_duration,
                        left=task.start_time,
                        height=0.6,
                        color=LLMPlotGenerator.PREFILL_COLOR,
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=0.5,
                        label="Prefill" if y_pos == 0 else "",
                    )

                    # Add task label for prefill
                    if max_tasks is not None and max_tasks <= 50:
                        ax.text(
                            task.start_time + prefill_duration / 2,
                            y_pos,
                            f"{task.task_id}\n(P)",
                            ha="center",
                            va="center",
                            fontsize=6,
                        )

                # Plot decode phase (prefill_completion_time to completion_time)
                if task.prefill_completion_time is not None and task.completion_time is not None:
                    decode_duration = task.completion_time - task.prefill_completion_time
                    if decode_duration > 0:  # Only plot if there's actual decode time
                        ax.barh(
                            y_pos,
                            decode_duration,
                            left=task.prefill_completion_time,
                            height=0.6,
                            color=LLMPlotGenerator.DECODE_COLOR,
                            alpha=0.8,
                            edgecolor="black",
                        linewidth=0.5,
                        label="Decode" if y_pos == 0 else "",
                    )

                    # Add task label for decode
                    if max_tasks is not None and max_tasks <= 50:
                        ax.text(
                            task.prefill_completion_time + decode_duration / 2,
                            y_pos,
                            f"{task.task_id}\n(D)",
                            ha="center",
                            va="center",
                            fontsize=6,
                        )

            y_pos += 1

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time (sec)", fontweight="bold")
        ax.set_ylabel("GPU Group", fontweight="bold")
        task_count_str = "all" if max_tasks is None else str(max_tasks)
        ax.set_title(f"Token-Level Schedule Gantt Chart ({task_count_str} tasks)", fontsize=14, fontweight="bold")

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=LLMPlotGenerator.PREFILL_COLOR, label="Prefill (Compute-bound)", alpha=0.8),
            Patch(facecolor=LLMPlotGenerator.DECODE_COLOR, label="Decode (Memory-bound)", alpha=0.8),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved token Gantt chart to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_batch_efficiency(
        results: Dict[str, Dict],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot batch efficiency comparison across algorithms.

        Args:
            results: Dictionary mapping algorithm name to metrics dict
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        algorithms = []
        batch_efficiencies = []
        throughputs = []

        for algo_name, result in results.items():
            if result.get("success", False):
                metrics = result["metrics"]
                algorithms.append(algo_name)
                batch_efficiencies.append(metrics.get("batch_efficiency", 0.0))
                throughputs.append(metrics.get("throughput", 0.0))

        if not algorithms:
            print("No successful results to plot")
            return

        x = np.arange(len(algorithms))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Batch efficiency
        bars1 = ax1.bar(
            x,
            batch_efficiencies,
            color="#51cf66",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_xlabel("Algorithm", fontweight="bold")
        ax1.set_ylabel("Batch Efficiency", fontweight="bold")
        ax1.set_title("Batch Efficiency Comparison", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars1, batch_efficiencies):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Throughput
        bars2 = ax2.bar(
            x,
            throughputs,
            color="#4dabf7",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_xlabel("Algorithm", fontweight="bold")
        ax2.set_ylabel("Throughput (tasks/sec)", fontweight="bold")
        ax2.set_title("Throughput Comparison", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars2, throughputs):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved batch efficiency plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_task_completion_timeline(
        tasks: List[LLMTask],
        save_path: Optional[str] = None,
        show: bool = False,
        max_tasks: int = 100,
    ) -> None:
        """
        Plot task completion time vs arrival time.

        Args:
            tasks: List of LLM tasks with scheduling information
            save_path: Path to save the figure
            show: Whether to display the plot
            max_tasks: Maximum number of tasks to display
        """
        scheduled_tasks = [t for t in tasks if t.is_scheduled()][:max_tasks]

        if not scheduled_tasks:
            print("No scheduled tasks to plot")
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        task_ids = [t.task_id for t in scheduled_tasks]
        arrival_times = [t.arrival_time for t in scheduled_tasks]
        completion_times = [t.completion_time for t in scheduled_tasks]
        weights = [t.weight for t in scheduled_tasks]

        # Color by weight
        norm = plt.Normalize(min(weights), max(weights))
        cmap = plt.cm.viridis
        colors = [cmap(norm(w)) for w in weights]

        x = np.arange(len(task_ids))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            arrival_times,
            width,
            label="Arrival Time",
            color="#ffd43b",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            completion_times,
            width,
            label="Completion Time",
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xlabel("Task ID", fontweight="bold")
        ax.set_ylabel("Time (sec)", fontweight="bold")
        ax.set_title("Task Arrival vs Completion Time", fontsize=14, fontweight="bold")
        ax.set_xticks(x[::5])  # Show every 5th task label
        ax.set_xticklabels(task_ids[::5], rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add colorbar for weight
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Task Weight", fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved task completion timeline to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_phase_distribution(
        tasks: List[LLMTask],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot distribution of prefill vs decode phases.

        Args:
            tasks: List of LLM tasks
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        prefill_tasks = [t for t in tasks if t.phase == Phase.PREFILL]
        decode_tasks = [t for t in tasks if t.phase == Phase.DECODE]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Phase count pie chart
        phase_counts = {
            "Prefill": len(prefill_tasks),
            "Decode": len(decode_tasks),
        }

        colors_pie = [LLMPlotGenerator.PREFILL_COLOR, LLMPlotGenerator.DECODE_COLOR]
        wedges, texts, autotexts = ax1.pie(
            phase_counts.values(),
            labels=phase_counts.keys(),
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            wedgeprops={"edgecolor": "black", "linewidth": 0.5},
        )
        ax1.set_title("Task Phase Distribution", fontsize=12, fontweight="bold")

        # Token count distribution
        prefill_tokens = [t.tokens for t in prefill_tasks]
        decode_tokens = [t.tokens for t in decode_tasks]

        ax2.hist(
            prefill_tokens,
            bins=20,
            alpha=0.6,
            label="Prefill Tokens",
            color=LLMPlotGenerator.PREFILL_COLOR,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.hist(
            decode_tokens,
            bins=20,
            alpha=0.6,
            label="Decode Tokens",
            color=LLMPlotGenerator.DECODE_COLOR,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_xlabel("Token Count", fontweight="bold")
        ax2.set_ylabel("Frequency", fontweight="bold")
        ax2.set_title("Token Count Distribution by Phase", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved phase distribution plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def generate_all_plots(
        results: Dict[str, Dict],
        tasks_per_algorithm: Dict[str, List[LLMTask]],
        cluster: LLMCluster,
        output_dir: str,
        experiment_name: str,
    ) -> None:
        """
        Generate all visualization plots for an experiment.

        Args:
            results: Dictionary mapping algorithm name to results dict
            tasks_per_algorithm: Dictionary mapping algorithm name to scheduled tasks
            cluster: LLM cluster object
            output_dir: Output directory path
            experiment_name: Name of the experiment
        """
        output_path = Path(output_dir) / "figures"
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating visualizations in {output_path}...")

        # 1. Algorithm comparison
        LLMPlotGenerator.plot_algorithm_comparison(
            results,
            cluster,
            save_path=str(output_path / f"{experiment_name}_comparison.png"),
            show=False,
        )

        # 2. GPU utilization
        LLMPlotGenerator.plot_gpu_utilization(
            results,
            cluster,
            save_path=str(output_path / f"{experiment_name}_gpu_utilization.png"),
            show=False,
        )

        # 3. Batch efficiency and throughput
        LLMPlotGenerator.plot_batch_efficiency(
            results,
            save_path=str(output_path / f"{experiment_name}_batch_efficiency.png"),
            show=False,
        )

        # 4. Token Gantt charts for each algorithm
        for algo_name, tasks in tasks_per_algorithm.items():
            # Full Gantt chart (limited tasks)
            LLMPlotGenerator.plot_token_gantt_chart(
                tasks,
                cluster,
                save_path=str(output_path / f"{experiment_name}_gantt_{algo_name}.png"),
                show=False,
                max_tasks=30,
            )

        # 5. Task completion timeline for best algorithm
        if tasks_per_algorithm:
            best_algo = min(
                [k for k in results.keys() if results[k].get("success", False)],
                key=lambda k: results[k]["metrics"]["weighted_waiting_time"]
            )
            LLMPlotGenerator.plot_task_completion_timeline(
                tasks_per_algorithm[best_algo],
                save_path=str(output_path / f"{experiment_name}_timeline_{best_algo}.png"),
                show=False,
                max_tasks=50,
            )

        # 6. Phase distribution (use tasks from first algorithm)
        if tasks_per_algorithm:
            first_algo = list(tasks_per_algorithm.keys())[0]
            LLMPlotGenerator.plot_phase_distribution(
                tasks_per_algorithm[first_algo],
                save_path=str(output_path / f"{experiment_name}_phase_distribution.png"),
                show=False,
            )

        print(f"Visualization generation complete!")
