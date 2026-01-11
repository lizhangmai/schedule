"""
事件驱动仿真框架
"""

import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(Enum):
    """事件类型枚举"""
    TASK_ARRIVAL = "task_arrival"
    TASK_COMPLETION = "task_completion"


@dataclass
class Event:
    """
    仿真事件

    属性:
        timestamp: float - 事件发生时间
        event_type: EventType - 事件类型
        data: Any - 事件关联的数据（通常是 Task 对象）
    """

    timestamp: float
    event_type: EventType
    data: Any = None

    def __lt__(self, other: "Event") -> bool:
        """支持优先队列排序（按时间戳）"""
        return self.timestamp < other.timestamp

    def __repr__(self) -> str:
        data_info = f"({self.data})" if self.data else ""
        return f"Event({self.event_type.value}, t={self.timestamp:.2f}{data_info})"


class EventQueue:
    """
    事件优先队列

    按时间戳从小到大处理事件，实现事件驱动仿真
    """

    def __init__(self):
        self._queue: List[Event] = []
        self._event_handlers: Dict[EventType, List[Callable]] = {}

    def push(self, event: Event) -> None:
        """
        添加事件到队列

        Args:
            event: 事件对象
        """
        heapq.heappush(self._queue, event)

    def pop(self) -> Optional[Event]:
        """
        取出下一个事件（时间戳最小）

        Returns:
            事件对象，如果队列为空返回 None
        """
        if not self._queue:
            return None
        return heapq.heappop(self._queue)

    def peek(self) -> Optional[Event]:
        """
        查看下一个事件但不移除

        Returns:
            事件对象，如果队列为空返回 None
        """
        if not self._queue:
            return None
        return self._queue[0]

    def is_empty(self) -> bool:
        """判断队列是否为空"""
        return len(self._queue) == 0

    def size(self) -> int:
        """获取队列大小"""
        return len(self._queue)

    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        注册事件处理器

        Args:
            event_type: 事件类型
            handler: 处理函数，接收 event 作为参数
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def dispatch(self, event: Event) -> None:
        """
        分发事件到对应的处理器

        Args:
            event: 事件对象
        """
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            handler(event)

    def clear(self) -> None:
        """清空队列"""
        self._queue.clear()

    def __repr__(self) -> str:
        return f"EventQueue(size={self.size()})"
