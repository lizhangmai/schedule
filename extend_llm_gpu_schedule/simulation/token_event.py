"""
Token-level events for LLM simulation.

Defines event types for fine-grained token-level scheduling.
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Any
from enum import Enum
import heapq

from ..models.llm_task import Phase

if TYPE_CHECKING:
    from ..models.llm_task import LLMTask
    from ..models.gpu_group import GPUGroup


class TokenType(Enum):
    """
    Token-level event types for LLM simulation.

    Unlike traditional task scheduling, LLM inference requires
    token-level granularity for:
    - Preemption at token boundaries
    - Continuous batching
    - Phase transitions (prefill -> decode)
    """
    TASK_ARRIVAL = "task_arrival"           # Task arrives at system
    PREFILL_START = "prefill_start"         # Prefill phase starts
    PREFILL_TOKEN = "prefill_token"         # Process prefill tokens
    PREFILL_COMPLETE = "prefill_complete"   # Prefill phase completes
    DECODE_START = "decode_start"           # Decode phase starts
    DECODE_TOKEN = "decode_token"           # Process decode tokens
    BATCH_DECODE = "batch_decode"           # Batch decode step
    TASK_COMPLETE = "task_complete"         # Task fully completes


@dataclass
class TokenEvent:
    """
    Token-level event for fine-grained LLM scheduling.

    Attributes:
        timestamp: Event time
        event_type: TokenType enum
        task: LLMTask reference (can be None for system events)
        gpu_group: GPUGroup processing the event (can be None)
        token_count: Number of tokens in this batch
        phase: Current phase (PREFILL or DECODE)
        priority: Event priority (higher = more important)
        metadata: Additional event data
    """
    timestamp: float
    event_type: TokenType
    task: Optional["LLMTask"] = None
    gpu_group: Optional["GPUGroup"] = None
    token_count: int = 0
    phase: Optional[Phase] = None
    priority: int = 0
    metadata: dict = field(default_factory=dict)

    def __lt__(self, other: "TokenEvent") -> bool:
        """
        Compare events for priority queue ordering.

        Primary: timestamp (earlier first)
        Secondary: priority (higher first)
        Tertiary: event_type (certain types take precedence)
        """
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp

        if self.priority != other.priority:
            return self.priority > other.priority

        # Event type precedence
        precedence = {
            TokenType.TASK_COMPLETE: 0,
            TokenType.PREFILL_COMPLETE: 1,
            TokenType.BATCH_DECODE: 2,
            TokenType.DECODE_TOKEN: 3,
            TokenType.PREFILL_TOKEN: 4,
            TokenType.DECODE_START: 5,
            TokenType.PREFILL_START: 6,
            TokenType.TASK_ARRIVAL: 7,
        }

        return precedence.get(self.event_type, 99) < precedence.get(other.event_type, 99)

    def __repr__(self) -> str:
        """String representation."""
        task_info = f"task={self.task.task_id}" if self.task else "no_task"
        gpu_info = f"gpu={self.gpu_group.group_id}" if self.gpu_group else "no_gpu"
        return (f"TokenEvent(t={self.timestamp:.4f}, type={self.event_type.value}, "
                f"{task_info}, {gpu_info}, tokens={self.token_count})")


class TokenEventQueue:
    """
    Priority queue for token-level events.

    Manages events in timestamp order with priority support.
    """

    def __init__(self):
        """Initialize empty event queue."""
        self._events: list[TokenEvent] = []
        self._event_count = 0

    def push(self, event: TokenEvent) -> None:
        """
        Push event onto the queue.

        Args:
            event: TokenEvent to add
        """
        heapq.heappush(self._events, event)
        self._event_count += 1

    def pop(self) -> Optional[TokenEvent]:
        """
        Pop next event from the queue.

        Returns:
            TokenEvent with lowest timestamp, or None if empty
        """
        if not self._events:
            return None
        self._event_count -= 1
        return heapq.heappop(self._events)

    def peek(self) -> Optional[TokenEvent]:
        """
        Peek at next event without removing it.

        Returns:
            TokenEvent with lowest timestamp, or None if empty
        """
        if not self._events:
            return None
        return self._events[0]

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._events) == 0

    def size(self) -> int:
        """Get number of events in queue."""
        return len(self._events)

    def clear(self) -> None:
        """Clear all events from queue."""
        self._events.clear()
        self._event_count = 0

    def get_events_by_time(self, time: float) -> list[TokenEvent]:
        """
        Get all events at or before a specific time.

        Args:
            time: Time threshold

        Returns:
            List of events (in order)
        """
        result = []
        while self._events and self._events[0].timestamp <= time:
            result.append(heapq.heappop(self._events))
            self._event_count -= 1
        return result

    def get_events_by_type(self, event_type: TokenType) -> list[TokenEvent]:
        """
        Get all events of a specific type.

        Note: This requires O(n) scan.

        Args:
            event_type: TokenType to filter

        Returns:
            List of matching events
        """
        return [e for e in self._events if e.event_type == event_type]

    def remove_events_for_task(self, task_id: str) -> int:
        """
        Remove all events for a specific task.

        Note: This requires O(n) scan and rebuild.

        Args:
            task_id: Task ID to remove

        Returns:
            Number of events removed
        """
        new_events = [e for e in self._events if e.task is None or e.task.task_id != task_id]
        removed = len(self._events) - len(new_events)
        self._events = new_events
        heapq.heapify(self._events)
        self._event_count = len(self._events)
        return removed

    def __len__(self) -> int:
        """Get queue size."""
        return len(self._events)

    def __repr__(self) -> str:
        """String representation."""
        return f"TokenEventQueue(size={len(self._events)}, next_t={self._events[0].timestamp if self._events else None})"


def create_task_arrival_event(task: "LLMTask") -> TokenEvent:
    """Create a task arrival event."""
    return TokenEvent(
        timestamp=task.arrival_time,
        event_type=TokenType.TASK_ARRIVAL,
        task=task,
        phase=task.phase,
        token_count=task.tokens,
    )


def create_prefill_start_event(
    task: "LLMTask",
    gpu_group: "GPUGroup",
    start_time: float
) -> TokenEvent:
    """Create a prefill start event."""
    return TokenEvent(
        timestamp=start_time,
        event_type=TokenType.PREFILL_START,
        task=task,
        gpu_group=gpu_group,
        phase=Phase.PREFILL,
        token_count=task.prefill_tokens,
    )


def create_prefill_complete_event(
    task: "LLMTask",
    gpu_group: "GPUGroup",
    completion_time: float
) -> TokenEvent:
    """Create a prefill complete event."""
    return TokenEvent(
        timestamp=completion_time,
        event_type=TokenType.PREFILL_COMPLETE,
        task=task,
        gpu_group=gpu_group,
        phase=Phase.PREFILL,
        token_count=task.prefill_tokens,
        priority=10,  # High priority for phase transitions
    )


def create_decode_start_event(
    task: "LLMTask",
    gpu_group: "GPUGroup",
    start_time: float
) -> TokenEvent:
    """Create a decode start event."""
    return TokenEvent(
        timestamp=start_time,
        event_type=TokenType.DECODE_START,
        task=task,
        gpu_group=gpu_group,
        phase=Phase.DECODE,
        token_count=task.decode_tokens,
    )


def create_decode_token_event(
    task: "LLMTask",
    gpu_group: "GPUGroup",
    timestamp: float,
    token_count: int = 1
) -> TokenEvent:
    """Create a decode token event."""
    return TokenEvent(
        timestamp=timestamp,
        event_type=TokenType.DECODE_TOKEN,
        task=task,
        gpu_group=gpu_group,
        phase=Phase.DECODE,
        token_count=token_count,
    )


def create_batch_decode_event(
    gpu_group: "GPUGroup",
    timestamp: float,
    token_count: int,
    tasks: list["LLMTask"]
) -> TokenEvent:
    """Create a batch decode event."""
    return TokenEvent(
        timestamp=timestamp,
        event_type=TokenType.BATCH_DECODE,
        task=tasks[0] if tasks else None,  # Representative task
        gpu_group=gpu_group,
        phase=Phase.DECODE,
        token_count=token_count,
        priority=5,  # Medium priority
        metadata={"num_tasks": len(tasks), "task_ids": [t.task_id for t in tasks]},
    )


def create_task_complete_event(
    task: "LLMTask",
    gpu_group: "GPUGroup",
    completion_time: float
) -> TokenEvent:
    """Create a task complete event."""
    return TokenEvent(
        timestamp=completion_time,
        event_type=TokenType.TASK_COMPLETE,
        task=task,
        gpu_group=gpu_group,
        phase=Phase.DECODE,  # Tasks end in decode
        token_count=0,
        priority=10,  # Highest priority
    )


if __name__ == "__main__":
    # Example usage
    print("Token Event Examples:")
    print("=" * 70)

    from ..models.llm_task import create_prefill_task

    # Create event queue
    queue = TokenEventQueue()

    # Create some test events
    task1 = create_prefill_task("T1", "Qwen3", 30, 1.0, 0.0, 2048, 2)
    task2 = create_prefill_task("T2", "Qwen3", 30, 1.0, 0.5, 1024, 2)

    # Add events out of order
    queue.push(create_task_arrival_event(task1))
    queue.push(create_task_arrival_event(task2))

    # Create mock GPU group
    class MockGPU:
        def __init__(self, gpu_id, model):
            self.gpu_id = gpu_id
            self.model = model

    class MockGPUGroup:
        def __init__(self, group_id):
            self.group_id = group_id
            self.gpus = []
            self.gpu_model = "H100"

    gpu_group = MockGPUGroup("group_1")

    # Add more events
    queue.push(create_prefill_start_event(task1, gpu_group, 0.0))
    queue.push(create_prefill_complete_event(task1, gpu_group, 0.5))
    queue.push(TokenEvent(timestamp=1.0, event_type=TokenType.TASK_COMPLETE, task=task1))

    print(f"\nEvent Queue: {queue}")
    print(f"\nProcessing events in order:")

    while not queue.is_empty():
        event = queue.pop()
        print(f"  t={event.timestamp:6.3f}: {event.event_type.value:20} - {event}")

    print("\n" + "=" * 70)
