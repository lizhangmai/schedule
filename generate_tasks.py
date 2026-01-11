"""
任务数据生成器

生成带有合理 deadline 的任务数据，支持多种负载级别

使用示例:
    # 生成指定编号的数据集（中等负载）
    python generate_tasks.py --task-id 3

    # 生成多个数据集
    python generate_tasks.py --task-ids 3,4,5

    # 使用指定种子生成数据集
    python generate_tasks.py --task-id 3 --seed 456

    # 指定负载级别
    python generate_tasks.py --task-id 3 --load-level high

    # 生成高负载数据集
    python generate_tasks.py --task-ids 3,4,5 --load-level extreme

负载级别说明:
    low      - 低负载，宽松 deadline，任务稀疏
    medium   - 中等负载，默认设置
    high     - 高负载，紧张 deadline，任务密集
    extreme  - 极高负载，非常紧张的 deadline，需要多GPU并行
"""

import argparse
import random
import re
from pathlib import Path
from typing import Literal


# 负载级别配置
LOAD_LEVEL_CONFIGS = {
    "low": {
        "num_tasks": 500,
        "workload_range": (50, 400),
        "arrival_time_range": (0, 5000),
        "deadline_factor_range": (3.0, 5.0),
        "weight_range": (1, 5),
        "description": "低负载：宽松 deadline，任务稀疏",
    },
    "medium": {
        "num_tasks": 1000,
        "workload_range": (50, 800),
        "arrival_time_range": (0, 2500),
        "deadline_factor_range": (1.5, 3.0),
        "weight_range": (1, 5),
        "description": "中等负载：默认设置",
    },
    "high": {
        "num_tasks": 1500,
        "workload_range": (100, 800),
        "arrival_time_range": (0, 1500),
        "deadline_factor_range": (0.8, 1.5),
        "weight_range": (1, 5),
        "description": "高负载：紧张 deadline，任务密集",
    },
    "extreme": {
        "num_tasks": 2000,
        "workload_range": (200, 800),
        "arrival_time_range": (0, 1000),
        "deadline_factor_range": (0.5, 1.0),
        "weight_range": (1, 5),
        "description": "极高负载：非常紧张的 deadline，必须多GPU并行",
    },
}


def generate_tasks(
    num_tasks: int = 1000,
    seed: int = 42,
    workload_range: tuple = (50, 800),
    memory_options: list[int] | None = None,
    arrival_time_range: tuple = (0, 2500),
    deadline_factor_range: tuple = (1.5, 3.0),
    weight_range: tuple = (1, 5),
    baseline_scaling_factor: float = 30.0,
) -> list:
    """
    生成任务列表

    Args:
        num_tasks: 任务数量
        seed: 随机种子
        workload_range: 工作量范围 (min, max)
        memory_options: 显存需求选项 (GB)
        arrival_time_range: 到达时间范围
        deadline_factor_range: deadline 因子范围
            deadline = arrival + execution_time * factor
            factor >= 1.0 表示有一定时间余量
        weight_range: 权重范围
        baseline_scaling_factor: 基准 GPU 的 scaling_factor (用于计算执行时间)

    Returns:
        任务字典列表
    """
    random.seed(seed)

    if memory_options is None:
        memory_options = [6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    tasks = []
    arrival_times = sorted(random.uniform(*arrival_time_range) for _ in range(num_tasks))

    for i, arrival_time in enumerate(arrival_times, 1):
        workload = random.randint(*workload_range)
        memory = random.choice(memory_options)
        weight = random.randint(*weight_range)

        # 计算在基准 GPU 上的执行时间
        baseline_exec_time = workload / baseline_scaling_factor

        # 设置 deadline，给予一定的时间余量
        deadline_factor = random.uniform(*deadline_factor_range)
        deadline = arrival_time + baseline_exec_time * deadline_factor

        tasks.append({
            "Task": f"T{i}",
            "Workload": workload,
            "Memory": memory,
            "Deadline": round(deadline, 2),
            "Weight": weight,
            "ArrivalTime": round(arrival_time, 2),
        })

    return tasks


def save_tasks_to_csv(tasks: list, output_path: str | Path) -> None:
    """
    将任务列表保存到 CSV 文件

    Args:
        tasks: 任务字典列表
        output_path: 输出文件路径
    """
    import pandas as pd

    df = pd.DataFrame(tasks)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(tasks)} tasks and saved to {output_path}")


def get_existing_task_ids(data_dir: Path) -> set[int]:
    """获取已存在的数据集编号"""
    existing_ids = set()
    for file in data_dir.glob("tasks*.csv"):
        match = re.search(r"tasks(\d+)\.csv", file.name)
        if match:
            existing_ids.add(int(match.group(1)))
    return existing_ids


def get_next_task_id(data_dir: Path) -> int:
    """获取下一个可用的数据集编号"""
    existing_ids = get_existing_task_ids(data_dir)
    if not existing_ids:
        return 1
    return max(existing_ids) + 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成任务数据集，支持多种负载级别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --task-id 3                      # 生成 tasks3.csv (中等负载)
  %(prog)s --task-id 3 --load-level high    # 生成高负载 tasks3.csv
  %(prog)s --task-ids 3,4,5 --load-level extreme  # 生成极高负载数据集
  %(prog)s --all --load-level high          # 生成所有缺失的高负载数据集

负载级别: low, medium, high, extreme
        """
    )
    parser.add_argument("--task-id", type=int, help="生成单个数据集编号")
    parser.add_argument("--task-ids", type=str, help="生成多个数据集编号，逗号分隔，如 3,4,5")
    parser.add_argument("--seed", type=int, default=42, help="随机种子基准值（默认: 42）")
    parser.add_argument("--all", action="store_true", help="自动生成所有缺失的数据集")
    parser.add_argument(
        "--load-level",
        type=str,
        choices=["low", "medium", "high", "extreme"],
        default="medium",
        help="负载级别（默认: medium）",
    )

    args = parser.parse_args()

    # 创建 data 目录
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    # 获取负载配置
    load_config = LOAD_LEVEL_CONFIGS[args.load_level]

    # 确定要生成哪些数据集
    task_ids = []
    if args.all:
        # 扫描 data/ 目录，生成缺失的
        existing = get_existing_task_ids(data_dir)
        max_id = max(existing) if existing else 0
        task_ids = [i for i in range(1, max_id + 6) if i not in existing]
        if not task_ids:
            print("All datasets already exist. Use --task-id to generate a specific one.")
            return
    elif args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]
    elif args.task_id:
        task_ids = [args.task_id]
    else:
        # 默认生成下一个编号
        task_ids = [get_next_task_id(data_dir)]

    # 生成数据集
    print(f"Data directory: {data_dir}")
    print(f"Load level: {args.load_level} - {load_config['description']}")
    print()

    for task_id in sorted(task_ids):
        # 每个数据集使用不同的种子
        seed = args.seed + task_id * 100

        filename = f"tasks{task_id}.csv"
        filepath = data_dir / filename

        print(f"Generating {filename}...")
        tasks = generate_tasks(
            num_tasks=load_config["num_tasks"],
            seed=seed,
            workload_range=load_config["workload_range"],
            arrival_time_range=load_config["arrival_time_range"],
            deadline_factor_range=load_config["deadline_factor_range"],
            weight_range=load_config["weight_range"],
        )
        save_tasks_to_csv(tasks, filepath)

    print()
    print("Task generation complete!")
    print()
    print("Dataset files saved to:", data_dir)
    print()
    print("Load level configurations:")
    for level, config in LOAD_LEVEL_CONFIGS.items():
        print(f"  {level:8s}: {config['description']}")
    print()
    print("Deadline configuration:")
    print(f"  - Baseline GPU: A30 (scaling_factor={30.0})")
    print(f"  - Current load level {args.load_level}: deadline_factor_range = {load_config['deadline_factor_range']}")


if __name__ == "__main__":
    main()
