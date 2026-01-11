"""
CSV 数据加载器
"""

from pathlib import Path
from typing import List

import pandas as pd

from ..models.task import Task


def load_tasks_from_csv(file_path: str | Path) -> List[Task]:
    """
    从 CSV 文件加载任务

    Args:
        file_path: CSV 文件路径

    Returns:
        任务列表

    CSV 格式:
        Task,Workload,Memory,Deadline,Weight,ArrivalTime
        T1,211,24,17.98,4,0.36
        ...
    """
    df = pd.read_csv(file_path)

    tasks = []
    for _, row in df.iterrows():
        task = Task(
            task_id=str(row["Task"]),
            workload=float(row["Workload"]),
            memory=int(row["Memory"]),
            deadline=float(row["Deadline"]),
            weight=int(row["Weight"]),
            arrival_time=float(row["ArrivalTime"]),
        )
        tasks.append(task)

    return tasks


def load_dataset(dataset_name: str, data_dir: str | Path = None) -> List[Task]:
    """
    加载指定的数据集

    Args:
        dataset_name: 数据集名称 (如 "tasks1.csv")
        data_dir: 数据目录，默认为当前目录下的 data 文件夹

    Returns:
        任务列表
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    file_path = data_dir / dataset_name
    return load_tasks_from_csv(file_path)
