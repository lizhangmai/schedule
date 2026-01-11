"""
CSV 数据加载器
"""

import logging
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

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: CSV 格式或数据无效
    """
    # 读取 CSV 文件
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Task file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV {file_path}: {e}")

    # 验证必需的列
    required_columns = ["Task", "Workload", "Memory", "Deadline", "Weight", "ArrivalTime"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 检查缺失值
    if df[required_columns].isnull().any().any():
        nan_info = df[required_columns].isnull().sum()
        nan_columns = [col for col in required_columns if nan_info[col] > 0]
        raise ValueError(f"CSV contains missing values in columns: {nan_columns}")

    # 验证并转换每一行
    tasks = []
    for idx, row in df.iterrows():
        try:
            task_id = str(row["Task"])
            workload = float(row["Workload"])
            memory = int(row["Memory"])
            deadline = float(row["Deadline"])
            weight = int(row["Weight"])
            arrival_time = float(row["ArrivalTime"])

            # 业务逻辑验证
            if workload <= 0:
                raise ValueError(f"Row {idx + 1}: Workload must be positive, got {workload}")
            if memory <= 0:
                raise ValueError(f"Row {idx + 1}: Memory must be positive, got {memory}")
            if deadline < 0:
                raise ValueError(f"Row {idx + 1}: Deadline must be non-negative, got {deadline}")
            if weight <= 0:
                raise ValueError(f"Row {idx + 1}: Weight must be positive, got {weight}")
            if arrival_time < 0:
                raise ValueError(f"Row {idx + 1}: Arrival time must be non-negative, got {arrival_time}")

            # 警告但不阻止：deadline 早于 arrival_time
            if deadline < arrival_time:
                logging.warning(
                    f"Row {idx + 1} ({task_id}): Deadline ({deadline:.2f}) is before "
                    f"arrival time ({arrival_time:.2f}). Task will immediately miss deadline."
                )

            task = Task(
                task_id=task_id,
                workload=workload,
                memory=memory,
                deadline=deadline,
                weight=weight,
                arrival_time=arrival_time,
            )
            tasks.append(task)

        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data in row {idx + 1}: {e}")

    if not tasks:
        raise ValueError(f"No valid tasks found in {file_path}")

    logging.info(f"Loaded {len(tasks)} tasks from {file_path}")
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
