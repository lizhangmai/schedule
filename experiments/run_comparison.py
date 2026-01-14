"""
算法对比实验脚本

运行多种调度算法，生成对比报告和可视化结果
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.baseline.fifo_scheduler import FIFOScheduler
from src.algorithms.baseline.greedy import GreedyScheduler
from src.algorithms.baseline.sa_greedy import SAGreedyScheduler
from src.models.cluster import create_small_cluster, create_medium_cluster, create_large_cluster
from src.metrics.calculator import MetricsCalculator
from src.utils.data_loader import load_tasks_from_csv
from src.visualization.plots import PlotGenerator
from src.simulation.simulator import SimulationResult


def setup_logging(results_dir: str, experiment_name: str = None) -> None:
    """
    配置 logging 模块，将日志输出到文件和控制台

    Args:
        results_dir: 结果输出目录
        experiment_name: 实验名称（可选）
    """
    logs_dir = Path(results_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_filename = f"{experiment_name}_{timestamp}.log"
    else:
        log_filename = f"experiment_{timestamp}.log"

    log_file = logs_dir / log_filename

    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()  # 清除现有处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")


def run_experiment(
    tasks_file: str,
    cluster_size: str = "small",
    algorithms: List[str] = None,
) -> Dict[str, Tuple[SimulationResult, object]]:
    """
    运行单次实验

    Args:
        tasks_file: 任务数据文件路径
        cluster_size: 集群规模 (small/medium/large)
        algorithms: 要运行的算法列表

    Returns:
        Dict[算法名, (仿真结果, 集群对象)]
    """
    if algorithms is None:
        algorithms = ["FIFO", "Greedy", "SAGreedy"]

    # 加载任务
    tasks = load_tasks_from_csv(tasks_file)
    logging.info(f"Loaded {len(tasks)} tasks from {tasks_file}")

    # 集群工厂
    cluster_factories = {
        "small": create_small_cluster,
        "medium": create_medium_cluster,
        "large": create_large_cluster,
    }

    # 调度器工厂函数
    def create_scheduler(algo_name: str, cluster):
        """根据算法名称创建调度器"""
        if algo_name == "FIFO":
            return FIFOScheduler(cluster)
        elif algo_name == "Greedy":
            return GreedyScheduler(cluster)
        elif algo_name == "SAGreedy":
            return SAGreedyScheduler(cluster)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

    # 运行各算法 - 每个算法使用独立的集群和任务副本
    results = {}
    for algo_name in algorithms:
        logging.info(f"Running {algo_name}...")

        # 关键修复：每个算法创建新的集群实例
        cluster = cluster_factories[cluster_size]()
        logging.info(f"  Created {cluster_size} cluster with {cluster.get_gpu_count()} GPUs")

        # 关键修复：每个算法使用任务的副本，避免状态污染
        from copy import deepcopy
        tasks_copy = deepcopy(tasks)

        scheduler = create_scheduler(algo_name, cluster)
        scheduled_tasks = scheduler.schedule(tasks_copy)

        # 创建仿真结果（提取有效完成时间）
        completion_times = [t.completion_time for t in scheduled_tasks if t.completion_time is not None]
        makespan = max(completion_times) if completion_times else 0.0

        # 边界情况警告
        if makespan == 0 and scheduled_tasks:
            logging.warning(
                f"  {algo_name}: Makespan is 0 but {len(scheduled_tasks)} tasks exist. "
                "Check if tasks were properly scheduled."
            )
        elif makespan == 0 and not scheduled_tasks:
            logging.warning(f"  {algo_name}: No tasks were scheduled.")

        result = SimulationResult(
            tasks=scheduled_tasks,
            makespan=makespan,
            total_weighted_tardiness=sum(t.get_weighted_tardiness() for t in scheduled_tasks),
            metadata={"algorithm": algo_name},
        )

        # 保存结果和集群状态
        results[algo_name] = (result, cluster)

        # 计算并记录基本统计
        metrics = MetricsCalculator.calculate(scheduled_tasks, cluster, result)
        logging.info(f"  Makespan: {metrics.makespan:.2f}")
        logging.info(f"  Weighted Tardiness: {metrics.weighted_tardiness:.2f}")
        logging.info(f"  Deadline Miss Rate: {metrics.deadline_miss_rate:.2%}")
        logging.info(f"  GPU Time Util: {metrics.gpu_time_utilization:.2%}")
        logging.info(f"  Avg Concurrent Tasks: {metrics.gpu_average_concurrent_tasks:.2f}")
        logging.info(f"  Peak Memory Util: {metrics.gpu_peak_memory_utilization:.2%}")
        logging.info(f"  Avg Memory Util: {metrics.gpu_average_memory_utilization:.2%}")

    return results


def save_schedule_results(
    results: Dict[str, Tuple[SimulationResult, object]],
    save_path: Path,
) -> Dict:
    """
    保存完整调度结果到 JSON 文件

    保存内容包括：
    - 每个任务的具体调度信息（task_id, gpu_id, start_time, completion_time）
    - 算法的整体指标（makespan, weighted_tardiness）

    Args:
        results: Dict[算法名, (仿真结果, 集群对象)]
        save_path: 保存路径

    Returns:
        保存的数据字典
    """
    schedule_data = {}

    for algo_name, (result, cluster) in results.items():
        algo_data = {
            "makespan": result.makespan,
            "total_weighted_tardiness": result.total_weighted_tardiness,
            "scheduled_task_count": len([t for t in result.tasks if t.is_scheduled()]),
            "tasks": []
        }

        # 保存每个任务的调度信息
        for task in result.tasks:
            if task.is_scheduled():
                task_data = {
                    "task_id": task.task_id,
                    "gpu_id": task.assigned_gpu.gpu_id if task.assigned_gpu else None,
                    "start_time": task.start_time,
                    "completion_time": task.completion_time,
                    "workload": task.workload,
                    "memory": task.memory,
                    "deadline": task.deadline,
                    "weight": task.weight,
                    "arrival_time": task.arrival_time,
                    "is_deadline_missed": task.is_deadline_missed(),
                }
                algo_data["tasks"].append(task_data)

        schedule_data[algo_name] = algo_data

    # 保存到 JSON 文件
    with open(save_path, "w") as f:
        json.dump(schedule_data, f, indent=2)

    return schedule_data


def load_schedule_results(
    schedule_path: str,
    cluster_size: str = "small",
) -> Dict[str, Tuple[SimulationResult, object]]:
    """
    从 JSON 文件加载调度结果

    Args:
        schedule_path: 调度结果 JSON 文件路径
        cluster_size: 集群规模

    Returns:
        Dict[算法名, (仿真结果, 集群对象)]
    """
    with open(schedule_path, "r") as f:
        schedule_data = json.load(f)

    cluster_factories = {
        "small": create_small_cluster,
        "medium": create_medium_cluster,
        "large": create_large_cluster,
    }

    results = {}

    for algo_name, algo_data in schedule_data.items():
        # 创建新的集群实例
        cluster = cluster_factories[cluster_size]()

        # 重建 Task 对象列表
        from src.models.task import Task
        tasks = []
        for task_data in algo_data["tasks"]:
            task = Task(
                task_id=task_data["task_id"],
                workload=task_data["workload"],
                memory=task_data["memory"],
                deadline=task_data["deadline"],
                weight=task_data["weight"],
                arrival_time=task_data["arrival_time"],
            )
            # 恢复调度状态
            from src.models.gpu import GPU
            for gpu in cluster.gpus:
                if gpu.gpu_id == task_data["gpu_id"]:
                    task.assigned_gpu = gpu
                    break
            task.start_time = task_data["start_time"]
            task.completion_time = task_data["completion_time"]
            tasks.append(task)

        # 创建 SimulationResult
        result = SimulationResult(
            tasks=tasks,
            makespan=algo_data["makespan"],
            total_weighted_tardiness=algo_data["total_weighted_tardiness"],
            metadata={"algorithm": algo_name},
        )

        results[algo_name] = (result, cluster)

    return results


def save_results(
    results: Dict[str, Tuple[SimulationResult, object]],
    output_dir: str,
    experiment_name: str,
) -> None:
    """
    保存实验结果

    Args:
        results: Dict[算法名, (仿真结果, 集群对象)]
        output_dir: 输出目录
        experiment_name: 实验名称
    """
    output_path = Path(output_dir)

    # 创建子目录
    metrics_dir = output_path / "metrics"
    logs_dir = output_path / "logs"
    figures_dir = output_path / "figures"
    schedules_dir = output_path / "schedules"
    for dir_path in [metrics_dir, logs_dir, figures_dir, schedules_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving results for {experiment_name}...")

    # 1. 生成对比表格并保存到 metrics/ 子目录
    metrics_data = {}
    for algo_name, (result, cluster) in results.items():
        metrics = MetricsCalculator.calculate(result.tasks, cluster, result)
        metrics_data[algo_name] = metrics.to_dict()

    df = pd.DataFrame(metrics_data).T
    csv_path = metrics_dir / f"{experiment_name}_comparison.csv"
    df.to_csv(csv_path)
    logging.info(f"  Saved comparison table to {csv_path}")

    # 2. 生成指标 JSON 并保存到 metrics/ 子目录
    json_path = metrics_dir / f"{experiment_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logging.info(f"  Saved metrics JSON to {json_path}")

    # 3. 保存完整调度结果到 schedules/ 子目录
    schedule_path = schedules_dir / f"{experiment_name}_schedules.json"
    schedule_data = save_schedule_results(results, schedule_path)
    logging.info(f"  Saved schedule details to {schedule_path}")

    # 3. 生成可视化图表到 figures/ 子目录

    # 算法对比柱状图
    # Note: 此图表只绘制不依赖 GPU timeline 的指标（完成时间、deadline miss 等）
    # 因此使用 first_cluster 作为参数是安全的，这些指标只基于任务数据计算
    first_cluster = next(iter(results.values()))[1]
    comparison_path = figures_dir / f"{experiment_name}_comparison.png"
    PlotGenerator.plot_algorithm_comparison(
        {k: v for k, (v, _) in results.items()},
        first_cluster,
        save_path=str(comparison_path),
        show=False,
    )
    logging.info(f"  Saved comparison plot to {comparison_path}")

    # GPU 利用率对比 - 使用每个算法对应的集群
    algorithms = []
    time_utils = []
    avg_memory_utils = []

    for algo_name, (result, cluster) in results.items():
        metrics = MetricsCalculator.calculate(result.tasks, cluster, result)
        algorithms.append(algo_name)
        time_utils.append(metrics.gpu_time_utilization)
        avg_memory_utils.append(metrics.gpu_average_memory_utilization)

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, time_utils, width, label="Time Utilization", alpha=0.8)
    bars2 = ax.bar(x + width / 2, avg_memory_utils, width, label="Average Memory Utilization", alpha=0.8)

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
    util_path = figures_dir / f"{experiment_name}_gpu_utilization.png"
    plt.savefig(util_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"  Saved GPU utilization plot to {util_path}")

    # 为每个算法生成甘特图（两种：全部任务 + 前50个任务）
    for algo_name, (result, cluster) in results.items():
        # 全部任务的甘特图
        gantt_all_path = figures_dir / f"{experiment_name}_gantt_{algo_name}_all.png"
        PlotGenerator.plot_gantt_chart(
            result,
            cluster,
            save_path=str(gantt_all_path),
            show=False,
            max_tasks=None,  # 显示全部任务
        )
        logging.info(f"  Saved Gantt chart (all) for {algo_name}")

        # 前50个任务的甘特图
        gantt_first50_path = figures_dir / f"{experiment_name}_gantt_{algo_name}_first50.png"
        PlotGenerator.plot_gantt_chart(
            result,
            cluster,
            save_path=str(gantt_first50_path),
            show=False,
            max_tasks=50,  # 显示前50个任务
        )
        logging.info(f"  Saved Gantt chart (first50) for {algo_name}")


def plot_from_saved(
    schedule_path: str,
    cluster_size: str = "small",
    output_dir: str = None,
    experiment_name: str = None,
) -> None:
    """
    从保存的调度结果中生成图表（无需重新运行调度算法）

    Args:
        schedule_path: 调度结果 JSON 文件路径
        cluster_size: 集群规模
        output_dir: 输出目录（默认与原结果相同）
        experiment_name: 实验名称（默认从文件名提取）
    """
    # 加载调度结果
    results = load_schedule_results(schedule_path, cluster_size)

    # 确定输出目录
    if output_dir is None:
        output_dir = str(Path(schedule_path).parent.parent)  # results/
    if experiment_name is None:
        experiment_name = Path(schedule_path).stem  # 去掉 .json 后缀

    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Plotting from saved schedule: {schedule_path}")

    # 算法对比柱状图
    first_cluster = next(iter(results.values()))[1]
    comparison_path = figures_dir / f"{experiment_name}_comparison.png"
    PlotGenerator.plot_algorithm_comparison(
        {k: v for k, (v, _) in results.items()},
        first_cluster,
        save_path=str(comparison_path),
        show=False,
    )
    logging.info(f"  Saved comparison plot to {comparison_path}")

    # GPU 利用率对比
    algorithms = []
    time_utils = []
    avg_memory_utils = []

    for algo_name, (result, cluster) in results.items():
        metrics = MetricsCalculator.calculate(result.tasks, cluster, result)
        algorithms.append(algo_name)
        time_utils.append(metrics.gpu_time_utilization)
        avg_memory_utils.append(metrics.gpu_average_memory_utilization)

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width / 2, time_utils, width, label="Time Utilization", alpha=0.8)
    bars2 = ax.bar(x + width / 2, avg_memory_utils, width, label="Average Memory Utilization", alpha=0.8)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Utilization")
    ax.set_title("GPU Utilization Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.set_ylim([0, 1])

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
    util_path = figures_dir / f"{experiment_name}_gpu_utilization.png"
    plt.savefig(util_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"  Saved GPU utilization plot to {util_path}")

    # 为每个算法生成甘特图（两种：全部任务 + 前50个任务）
    for algo_name, (result, cluster) in results.items():
        # 全部任务的甘特图
        gantt_all_path = figures_dir / f"{experiment_name}_gantt_{algo_name}_all.png"
        PlotGenerator.plot_gantt_chart(
            result,
            cluster,
            save_path=str(gantt_all_path),
            show=False,
            max_tasks=None,  # 显示全部任务
        )
        logging.info(f"  Saved Gantt chart (all) for {algo_name}")

        # 前50个任务的甘特图
        gantt_first50_path = figures_dir / f"{experiment_name}_gantt_{algo_name}_first50.png"
        PlotGenerator.plot_gantt_chart(
            result,
            cluster,
            save_path=str(gantt_first50_path),
            show=False,
            max_tasks=50,  # 显示前50个任务
        )
        logging.info(f"  Saved Gantt chart (first50) for {algo_name}")

    logging.info("All plots generated from saved schedule data.")


def run_full_experiments() -> None:
    """运行完整实验矩阵"""
    # 自动扫描 data/ 目录获取所有数据集
    data_dir = Path(__file__).parent.parent / "data"
    dataset_files = sorted(data_dir.glob("tasks*.csv"))

    if not dataset_files:
        logging.warning(f"No datasets found in {data_dir}")
        return

    # 转换为相对路径格式 (data/tasksX.csv)
    datasets = [f"data/{f.name}" for f in dataset_files]

    cluster_sizes = ["small", "medium", "large"]
    algorithms = ["FIFO", "Greedy", "SAGreedy"]

    results_dir = Path(__file__).parent.parent / "results"

    for dataset in datasets:
        # 处理路径，支持 data/tasksX.csv 格式
        if dataset.startswith("data/"):
            dataset_name = dataset[5:]  # 去掉 "data/" 前缀
        else:
            dataset_name = dataset

        data_path = Path(__file__).parent.parent / dataset
        if not data_path.exists():
            logging.warning(f"{data_path} not found, skipping")
            continue

        for cluster_size in cluster_sizes:
            logging.info(f"{'='*60}")
            logging.info(f"Experiment: {dataset_name} - {cluster_size} cluster")
            logging.info(f"{'='*60}")

            results = run_experiment(
                str(data_path),
                cluster_size=cluster_size,
                algorithms=algorithms,
            )

            # 保存结果 (使用 dataset 的文件名，不含扩展名)
            experiment_name = f"{dataset_name[:-4]}_{cluster_size}"
            save_results(
                results,
                str(results_dir),
                experiment_name,
            )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPU Scheduling Experiment")
    parser.add_argument("--dataset", type=str, default="data/tasks1.csv",
                        help="Dataset file (e.g., data/tasks1.csv)")
    parser.add_argument("--cluster", type=str, default="small", choices=["small", "medium", "large"],
                        help="Cluster size")
    parser.add_argument("--algorithms", type=str, nargs="+",
                        default=["FIFO", "Greedy", "SAGreedy"],
                        help="Algorithms to run")
    parser.add_argument("--full", action="store_true", help="Run full experiment matrix")
    parser.add_argument("--plot-only", type=str, metavar="SCHEDULE_JSON",
                        help="Generate plots from saved schedule JSON file (no need to re-run SA)")

    args = parser.parse_args()

    # 设置日志
    results_dir = str(Path(__file__).parent.parent / "results")

    # 从保存的调度结果直接画图
    if args.plot_only:
        setup_logging(results_dir, "plot_only")
        plot_from_saved(args.plot_only, args.cluster)
        return

    if args.full:
        setup_logging(results_dir)
        run_full_experiments()
    else:
        # 处理数据集路径和名称
        dataset_path = args.dataset
        if not dataset_path.startswith("data/"):
            # 兼容旧格式，自动添加 data/ 前缀
            dataset_path = f"data/{args.dataset}"

        # 提取数据集名称（不含路径和扩展名）
        dataset_name = Path(dataset_path).stem  # 获取文件名不含扩展名
        experiment_name = f"{dataset_name}_{args.cluster}"

        setup_logging(results_dir, experiment_name)

        # 构造完整的数据文件路径
        data_path = Path(__file__).parent.parent / dataset_path
        if not data_path.exists():
            logging.error(f"Dataset file not found: {data_path}")
            logging.info(f"Use 'python generate_tasks.py' to generate datasets.")
            return

        results = run_experiment(str(data_path), args.cluster, args.algorithms)

        save_results(
            results,
            str(Path(__file__).parent.parent / "results"),
            experiment_name,
        )


if __name__ == "__main__":
    main()
