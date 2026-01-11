"""
任务数据生成器

生成带有合理 deadline 的任务数据

使用示例:
    # 生成指定编号的数据集
    python generate_tasks.py --task-id 3

    # 生成多个数据集
    python generate_tasks.py --task-ids 3,4,5

    # 使用指定种子生成数据集
    python generate_tasks.py --task-id 3 --seed 456

    # 自动生成所有缺失的数据集
    python generate_tasks.py --all
"""

import argparse
import random
import re
from pathlib import Path


def generate_tasks(
    num_tasks: int = 1000,
    seed: int = 42,
    workload_range: tuple = (50, 800),
    memory_options: list[int] | None = None,
    arrival_time_range: tuple = (0, 2500),
    deadline_factor_range: tuple = (1.5, 3.0),
    weight_range: tuple = (1, 5),
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

    Returns:
        任务字典列表
    """
    random.seed(seed)

    if memory_options is None:
        memory_options = [6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    tasks = []
    arrival_times = sorted(random.uniform(*arrival_time_range) for _ in range(num_tasks))

    # GPU 配置信息（用于计算合理的执行时间）
    # A100: scaling_factor=2.0, A30: scaling_factor=1.0, L40: scaling_factor=1.5
    # 使用 A30 作为基准（最慢的），这样 deadline 在所有 GPU 上都合理
    baseline_scaling_factor = 1.0

    for i, arrival_time in enumerate(arrival_times, 1):
        workload = random.randint(*workload_range)
        memory = random.choice(memory_options)
        weight = random.randint(*weight_range)

        # 计算在最慢 GPU 上的执行时间
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
        description="生成任务数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --task-id 3              # 生成 tasks3.csv
  %(prog)s --task-ids 3,4,5         # 生成 tasks3.csv, tasks4.csv, tasks5.csv
  %(prog)s --task-id 3 --seed 456   # 使用指定种子生成
  %(prog)s --all                    # 生成所有缺失的数据集
        """
    )
    parser.add_argument("--task-id", type=int, help="生成单个数据集编号")
    parser.add_argument("--task-ids", type=str, help="生成多个数据集编号，逗号分隔，如 3,4,5")
    parser.add_argument("--seed", type=int, default=42, help="随机种子基准值（默认: 42）")
    parser.add_argument("--all", action="store_true", help="自动生成所有缺失的数据集")

    args = parser.parse_args()

    # 创建 data 目录
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

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
    print()

    for task_id in sorted(task_ids):
        # 每个数据集使用不同的种子
        seed = args.seed + task_id * 100

        filename = f"tasks{task_id}.csv"
        filepath = data_dir / filename

        print(f"Generating {filename}...")
        tasks = generate_tasks(
            num_tasks=1000,
            seed=seed,
            workload_range=(50, 800),
            arrival_time_range=(0, 2500),
            deadline_factor_range=(1.5, 3.0),
        )
        save_tasks_to_csv(tasks, filepath)

    print()
    print("Task generation complete!")
    print()
    print("Dataset files saved to:", data_dir)
    print()
    print("Deadline configuration:")
    print("  - Baseline GPU: A30 (scaling_factor=1.0)")
    print("  - Deadline factor: 1.5-3.0x execution time")
    print("  - This means tasks should have reasonable time to complete")


if __name__ == "__main__":
    main()
