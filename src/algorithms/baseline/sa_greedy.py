"""
模拟退火调度器：结合模拟退火和贪心 GPU 分配

策略：
1. 以 Greedy 为搜索起点
2. 整理所有拖期任务，优先安排一定比例的拖期任务
   - 比例随温度下降：高温时优先安排较多拖期任务（更多探索），低温时优先安排较少（更多利用）
3. 在优先安排的拖期任务基础上，用 Greedy 形式安排其他任务
4. 计算评价指标 weighted_tardiness
5. 即使评价指标较差，也有一定概率接受（退火概率）
"""

import math
import random
from typing import List, Tuple, Optional, TYPE_CHECKING
from tqdm import tqdm

from ..base import BaseScheduler
from ...models.task import Task
from ...models.cluster import Cluster

if TYPE_CHECKING:
    from ...models.gpu import GPU


class SAGreedyScheduler(BaseScheduler):
    """
    模拟退火调度器：结合模拟退火和贪心 GPU 分配

    新策略：
    1. 以 Greedy 为搜索起点
    2. 整理所有拖期任务，优先安排一定比例的拖期任务
       - 比例随温度下降：高温时优先安排较多拖期任务（更多探索），低温时优先安排较少（更多利用）
    3. 在优先安排的拖期任务基础上，用 Greedy 形式安排其他任务
    4. 计算评价指标 weighted_tardiness
    5. 即使评价指标较差，也有一定概率接受（退火概率）

    参数：
    - initial_temp: 初始温度，默认 10000（需根据实际cost规模调整）
    - cooling_rate: 降温系数，默认 0.995
    - min_temp: 终止温度，默认 0.1
    - max_iterations: 最大迭代次数，默认 10000
    """

    def __init__(
        self,
        cluster: Cluster,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.8,
        min_temp: float = 1,
        max_iterations: int = 1000
    ):
        super().__init__()
        self.cluster = cluster
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations

    def schedule(self, tasks: List[Task]) -> List[Task]:
        """
        执行模拟退火调度

        Args:
            tasks: 待调度的任务列表

        Returns:
            调度后的任务列表
        """
        # 重置集群状态，确保从干净状态开始
        self.cluster.reset()

        # 运行模拟退火寻找最优任务顺序
        best_order = self._run_sa(tasks)

        # 重置集群，使用最优顺序进行实际调度
        self.cluster.reset()

        scheduled_tasks = []
        for task in best_order:
            if self._schedule_task_greedy(task):
                scheduled_tasks.append(task)

        self.scheduled_tasks = scheduled_tasks
        return self.scheduled_tasks

    def _run_sa(self, tasks: List[Task]) -> List[Task]:
        """
        运行模拟退火算法（新策略）

        新策略：
        1. 以 Greedy 为搜索起点
        2. 整理所有拖期任务，优先安排一定比例的拖期任务
           - 比例随温度下降：高温时优先安排较多拖期任务（更多探索），低温时优先安排较少（更多利用）
        3. 在优先安排的拖期任务基础上，用 Greedy 形式安排其他任务
        4. 计算评价指标 weighted_tardiness
        5. 即使评价指标较差，也有一定概率接受（退火概率）

        Args:
            tasks: 任务列表

        Returns:
            最优任务顺序
        """
        # 1. 以 Greedy 为搜索起点
        arrival_sorted = sorted(tasks, key=lambda t: (t.arrival_time, t.task_id))
        current_order = arrival_sorted.copy()
        current_cost, current_completions = self._evaluate_order(current_order)

        best_order = current_order.copy()
        best_cost = current_cost

        temp = self.initial_temp

        # 使用迭代次数而非温度阈值来控制循环，确保足够的搜索
        with tqdm(total=self.max_iterations, desc="模拟退火调度") as pbar:
            for _ in range(self.max_iterations):
                # 2. 生成新解：优先安排拖期任务
                new_order = self._generate_neighbor_with_tardy_priority(
                    current_order, current_completions, temp, self.initial_temp
                )

                # 3. 评估新解
                new_cost, new_completions = self._evaluate_order(new_order)

                # 5. 接受准则（即使评价指标较差，也有一定概率接受）
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    # 接受新解
                    current_order = new_order
                    current_cost = new_cost
                    current_completions = new_completions

                    # 更新最优解
                    if current_cost < best_cost:
                        best_order = current_order.copy()
                        best_cost = current_cost

                # 降温
                temp *= self.cooling_rate

                # 更新进度条显示信息
                pbar.set_postfix({
                    '温度': f'{temp:.1f}',
                    '当前cost': f'{current_cost:.1f}',
                    '最优cost': f'{best_cost:.1f}'
                })
                pbar.update(1)

        return best_order

    def _generate_neighbor_with_tardy_priority(
        self,
        tasks: List[Task],
        completions: dict[str, float],
        current_temp: float,
        initial_temp: float
    ) -> List[Task]:
        """
        生成新解：优先安排拖期任务

        新策略：
        1. 找出所有拖期任务
        2. 根据温度计算优先安排的拖期任务比例
           - 高温时比例高（更多探索），低温时比例低（更多利用）
        3. 将这些拖期任务提前到前面
        4. 其他任务按 arrival_time 排序（Greedy 方式）

        Args:
            tasks: 当前任务顺序
            completions: 任务ID -> 完成时间 的映射
            current_temp: 当前温度
            initial_temp: 初始温度

        Returns:
            新的任务顺序
        """
        # 1. 找出所有拖期任务（completion_time > deadline）
        tardy_tasks = []
        for task in tasks:
            completion_time = completions.get(task.task_id, float('inf'))
            if completion_time > task.deadline:
                tardiness = completion_time - task.deadline
                tardy_tasks.append((task, tardiness))

        # 2. 根据温度计算优先安排的拖期任务比例
        # 高温时比例高，低温时比例低
        # temp_ratio: 高温时接近 1，低温时接近 0
        temp_ratio = (current_temp - self.min_temp) / (initial_temp - self.min_temp)
        # 比例从 0.1（低温）到 0.8（高温）
        priority_ratio = min(0.8, max(0.1, temp_ratio))

        # 3. 选择拖期任务，按拖期量排序，选择比例最高的
        # 添加随机性：随机打乱拖期任务后再选择
        random.shuffle(tardy_tasks)
        tardy_tasks_sorted = sorted(tardy_tasks, key=lambda x: x[1], reverse=True)
        n_priority_tardy = max(1, int(len(tardy_tasks) * priority_ratio))
        priority_tardy_tasks = [t[0] for t in tardy_tasks_sorted[:n_priority_tardy]]

        # 4. 构建新顺序：
        #    a) 优先拖期任务按 arrival_time 排序
        #    b) 其他任务按 arrival_time 排序
        #    c) 合并
        #    d) 添加随机性：随机打乱优先拖期任务的顺序
        other_tasks = [t for t in tasks if t not in priority_tardy_tasks]

        # 根据温度调整随机性程度：高温时更多随机，低温时更接近 arrival_time 排序
        # 高温时 temp_ratio 接近 1（更多随机），低温时 temp_ratio 接近 0（更接近 arrival_time）
        randomness_level = temp_ratio

        # 随机决定拖期任务的排序方式
        if random.random() < 0.25:
            # 方式1：按拖期量排序
            priority_sorted = sorted(priority_tardy_tasks, key=lambda t: -(completions.get(t.task_id, float('inf')) - t.deadline))
        elif random.random() < 0.5:
            # 方式2：按 arrival_time 排序
            priority_sorted = sorted(priority_tardy_tasks, key=lambda t: (t.arrival_time, t.task_id))
        elif random.random() < 0.75:
            # 方式3：按 deadline 排序
            priority_sorted = sorted(priority_tardy_tasks, key=lambda t: (t.deadline, t.task_id))
        else:
            # 方式4：完全随机
            priority_sorted = priority_tardy_tasks.copy()
            random.shuffle(priority_sorted)

        # 随机决定其他任务的排序方式（基于温度调整随机性）
        if random.random() < randomness_level:
            # 随机打乱其他任务的内部顺序
            other_sorted = other_tasks.copy()
            random.shuffle(other_sorted)
        else:
            # 按 arrival_time 排序
            other_sorted = sorted(other_tasks, key=lambda t: (t.arrival_time, t.task_id))

        new_order = priority_sorted + other_sorted
        return new_order

    def _evaluate_order(self, task_order: List[Task]) -> Tuple[float, dict[str, float]]:
        """
        评估任务顺序的加权拖期（使用轻量级模拟，无需 deepcopy）

        Args:
            task_order: 任务顺序

        Returns:
            (加权拖期总和, 任务ID -> 完成时间 的映射)
        """
        total_weighted_tardiness = 0.0
        completions = {}  # 任务ID -> 完成时间

        # 轻量级跟踪：为每个GPU维护时间占用列表
        gpu_schedules = {gpu.gpu_id: [] for gpu in self.cluster.gpus}

        for task in task_order:
            best_gpu_id = None
            best_start_time = float('inf')
            best_completion_time = float('inf')

            # 找到能容纳该任务的GPU中最早的完成时间
            for gpu in self.cluster.gpus:
                if not gpu.can_accommodate(task):
                    continue

                # 找到最早的可用开始时间
                schedule = gpu_schedules[gpu.gpu_id]
                start_time = task.arrival_time
                execution_time = task.get_execution_time(gpu)

                # 收集候选开始时间点（到达时间 + 所有任务结束时间）
                candidate_times = [start_time]
                for s_start, s_end, _ in schedule:
                    if s_end >= start_time:
                        candidate_times.append(s_end)

                # 尝试每个候选时间，找到第一个不冲突的
                for candidate in candidate_times:
                    test_start = max(candidate, start_time)
                    test_end = test_start + execution_time

                    # 检查是否与所有已调度任务冲突
                    has_conflict = False
                    for s_start, s_end, _ in schedule:
                        # 冲突条件：时间段重叠
                        if not (test_end <= s_start or test_start >= s_end):
                            has_conflict = True
                            break

                    if not has_conflict:
                        start_time = test_start
                        break

                completion_time = start_time + execution_time

                if completion_time < best_completion_time:
                    best_completion_time = completion_time
                    best_start_time = start_time
                    best_gpu_id = gpu.gpu_id

            # 记录最佳调度
            if best_gpu_id is not None:
                gpu_schedules[best_gpu_id].append((best_start_time, best_completion_time, task))
                gpu_schedules[best_gpu_id].sort(key=lambda x: x[0])  # 保持有序以便后续查找
                completions[task.task_id] = best_completion_time

                # 计算加权拖期
                tardiness = max(0.0, best_completion_time - task.deadline)
                total_weighted_tardiness += task.weight * tardiness
            else:
                # 任务无法调度
                completions[task.task_id] = float('inf')

        return total_weighted_tardiness, completions

    def _find_best_gpu(self, cluster: Cluster, task: Task) -> Tuple[Optional["GPU"], float]:
        """
        为任务找到最佳 GPU（贪心策略：最早完成时间）

        Args:
            cluster: 集群对象
            task: 任务对象

        Returns:
            (最佳 GPU, 开始时间)
        """
        feasible_gpus = cluster.get_available_gpus(task)

        if not feasible_gpus:
            return None, 0.0

        best_gpu = None
        earliest_completion = float('inf')
        best_start_time = 0.0

        for gpu in feasible_gpus:
            start_time = max(task.arrival_time, gpu.find_earliest_start_time(task, task.arrival_time))
            completion_time = gpu.get_completion_time(task, start_time)

            if completion_time < earliest_completion:
                earliest_completion = completion_time
                best_start_time = start_time
                best_gpu = gpu

        return best_gpu, best_start_time

    def _schedule_task_greedy(self, task: Task) -> bool:
        """
        使用贪心策略调度单个任务

        Args:
            task: 任务对象

        Returns:
            是否成功调度
        """
        feasible_gpus = self.cluster.get_available_gpus(task)

        if not feasible_gpus:
            return False

        best_gpu = None
        earliest_completion = float('inf')
        best_start_time = 0.0

        for gpu in feasible_gpus:
            start_time = max(task.arrival_time, gpu.find_earliest_start_time(task, task.arrival_time))
            completion_time = gpu.get_completion_time(task, start_time)

            if completion_time < earliest_completion:
                earliest_completion = completion_time
                best_start_time = start_time
                best_gpu = gpu

        if best_gpu is not None:
            best_gpu.add_task(task, best_start_time)
            return True

        return False
