"""
LLM task model for inference scheduling.

Defines LLMTask representing an inference request with phase information
(Prefill vs Decode), token counts, and scheduling state.
"""

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .gpu_group import GPUGroup


class Phase(Enum):
    """
    LLM inference phases.

    Prefill (compute-bound): Processes the entire prompt at once.
    Decode (memory-bound): Generates tokens one at a time with KV cache.
    """
    PREFILL = "PREFILL"  # Prompt processing phase (compute-bound)
    DECODE = "DECODE"    # Token generation phase (memory-bound)

    @property
    def is_compute_bound(self) -> bool:
        """Check if phase is compute-bound."""
        return self == Phase.PREFILL

    @property
    def is_memory_bound(self) -> bool:
        """Check if phase is memory-bound."""
        return self == Phase.DECODE

    def __str__(self) -> str:
        return self.value


@dataclass
class LLMTask:
    """
    LLM inference task representing a single inference request.

    An LLM task has two phases:
    1. Prefill: Process the entire prompt (compute-bound, parallelizable)
    2. Decode: Generate output tokens (memory-bound, benefits from batching)

    Attributes:
        task_id: Unique task identifier (e.g., "T1", "T2")
        model_name: LLM model name (e.g., "DeepSeek-R1", "Qwen3")
        memory: Total model memory requirement in GB
        weight: Task priority weight (for weighted waiting time)
        arrival_time: When the request arrives
        phase: Initial phase (PREFILL or DECODE)
        tokens: Number of tokens to process in this phase
        tp_degree: Tensor parallelism degree (number of GPUs required)

    Computed fields:
        prefill_tokens: Total prefill tokens (for full request)
        decode_tokens: Total decode tokens (for full request)

    Scheduling state:
        assigned_gpu_group: List of GPUs allocated (multi-GPU)
        start_time: When prefill starts
        prefill_completion_time: When prefill completes
        completion_time: When all decode completes
        current_phase: Current execution phase
        remaining_tokens: Remaining tokens in current phase
    """
    task_id: str
    model_name: str
    memory: float  # GB (total memory requirement)
    weight: float
    arrival_time: float
    phase: Phase
    tokens: int
    tp_degree: int = 1  # Tensor parallelism degree

    # Computed fields (initialized in __post_init__)
    prefill_tokens: int = field(init=False)
    decode_tokens: int = field(init=False)

    # Scheduling state (initialized as None)
    assigned_gpu_group: Optional[List] = field(default=None, repr=False)
    start_time: Optional[float] = field(default=None, repr=False)
    prefill_completion_time: Optional[float] = field(default=None, repr=False)
    completion_time: Optional[float] = field(default=None, repr=False)
    current_phase: Optional[Phase] = field(default=None, repr=False)
    remaining_tokens: int = field(default=0, repr=False)

    def __post_init__(self):
        """Compute prefill and decode token counts from phase and tokens."""
        if self.phase == Phase.PREFILL:
            # This is a prefill-only task
            self.prefill_tokens = self.tokens
            self.decode_tokens = 0
        elif self.phase == Phase.DECODE:
            # This is a decode-only task (e.g., continuing generation)
            self.prefill_tokens = 0
            self.decode_tokens = self.tokens
        else:
            # Invalid phase
            raise ValueError(f"Invalid phase: {self.phase}")

        # Initialize remaining tokens
        self.remaining_tokens = self.tokens

    def get_waiting_time(self) -> float:
        """
        Calculate waiting time: completion_time - arrival_time.

        Returns:
            Waiting time in seconds, or 0 if not completed
        """
        if self.completion_time is None:
            return 0.0
        return self.completion_time - self.arrival_time

    def get_weighted_waiting_time(self) -> float:
        """
        Calculate weighted waiting time: weight * (completion - arrival).

        This is the primary optimization objective.

        Returns:
            Weighted waiting time, or 0 if not completed
        """
        return self.weight * self.get_waiting_time()

    def get_response_time(self) -> float:
        """
        Calculate response time: completion_time - start_time.

        Returns:
            Response time in seconds, or 0 if not completed
        """
        if self.start_time is None or self.completion_time is None:
            return 0.0
        return self.completion_time - self.start_time

    def get_queueing_time(self) -> float:
        """
        Calculate queueing time: start_time - arrival_time.

        Returns:
            Queueing time in seconds, or 0 if not started
        """
        if self.start_time is None:
            return 0.0
        return self.start_time - self.arrival_time

    def get_prefill_duration(self) -> float:
        """
        Get prefill phase duration.

        Returns:
            Prefill duration in seconds, or 0 if not completed
        """
        if (self.start_time is None or
            self.prefill_completion_time is None):
            return 0.0
        return self.prefill_completion_time - self.start_time

    def get_decode_duration(self) -> float:
        """
        Get decode phase duration.

        Returns:
            Decode duration in seconds, or 0 if not completed
        """
        if (self.prefill_completion_time is None or
            self.completion_time is None):
            return 0.0
        return self.completion_time - self.prefill_completion_time

    def is_scheduled(self) -> bool:
        """Check if task has been assigned GPUs."""
        return self.assigned_gpu_group is not None

    def is_started(self) -> bool:
        """Check if task has started execution."""
        return self.start_time is not None

    def is_completed(self) -> bool:
        """Check if task has completed all phases."""
        return self.completion_time is not None

    def is_in_prefill(self) -> bool:
        """Check if task is currently in prefill phase."""
        return self.current_phase == Phase.PREFILL

    def is_in_decode(self) -> bool:
        """Check if task is currently in decode phase."""
        return self.current_phase == Phase.DECODE

    def has_remaining_prefill(self) -> bool:
        """Check if task has remaining prefill tokens."""
        return self.remaining_tokens > 0 and self.current_phase in (None, Phase.PREFILL)

    def has_remaining_decode(self) -> bool:
        """Check if task has remaining decode tokens."""
        return self.remaining_tokens > 0 and self.current_phase == Phase.DECODE

    def transition_to_decode(self) -> None:
        """Transition task from prefill to decode phase."""
        if self.current_phase != Phase.PREFILL:
            raise ValueError(f"Cannot transition to decode from {self.current_phase}")

        self.current_phase = Phase.DECODE
        self.remaining_tokens = self.decode_tokens

    def reset(self) -> None:
        """Reset scheduling state for re-scheduling."""
        self.assigned_gpu_group = None
        self.start_time = None
        self.prefill_completion_time = None
        self.completion_time = None
        self.current_phase = None
        self.remaining_tokens = self.tokens

    def __lt__(self, other: "LLMTask") -> bool:
        """
        Compare tasks for priority queue ordering.

        Default ordering: by arrival time, then by weight (higher first).
        """
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        return self.weight > other.weight  # Higher weight first

    def __repr__(self) -> str:
        """String representation."""
        status = "pending"
        if self.is_completed():
            status = "completed"
        elif self.is_started():
            status = f"running_{self.current_phase.value}"

        return (f"LLMTask({self.task_id}, model={self.model_name}, "
                f"phase={self.phase.value}, tokens={self.tokens}, "
                f"weight={self.weight}, arrival={self.arrival_time:.2f}, "
                f"status={status})")


def create_prefill_task(
    task_id: str,
    model_name: str,
    memory: float,
    weight: float,
    arrival_time: float,
    tokens: int,
    tp_degree: int = 1
) -> LLMTask:
    """
    Create a prefill-only LLM task.

    Args:
        task_id: Unique task identifier
        model_name: LLM model name
        memory: Total memory requirement in GB
        weight: Task priority weight
        arrival_time: When the request arrives
        tokens: Number of prefill tokens
        tp_degree: Tensor parallelism degree

    Returns:
        LLMTask configured for prefill phase
    """
    return LLMTask(
        task_id=task_id,
        model_name=model_name,
        memory=memory,
        weight=weight,
        arrival_time=arrival_time,
        phase=Phase.PREFILL,
        tokens=tokens,
        tp_degree=tp_degree
    )


def create_decode_task(
    task_id: str,
    model_name: str,
    memory: float,
    weight: float,
    arrival_time: float,
    tokens: int,
    tp_degree: int = 1
) -> LLMTask:
    """
    Create a decode-only LLM task.

    Args:
        task_id: Unique task identifier
        model_name: LLM model name
        memory: Total memory requirement in GB
        weight: Task priority weight
        arrival_time: When the request arrives
        tokens: Number of decode tokens
        tp_degree: Tensor parallelism degree

    Returns:
        LLMTask configured for decode phase
    """
    return LLMTask(
        task_id=task_id,
        model_name=model_name,
        memory=memory,
        weight=weight,
        arrival_time=arrival_time,
        phase=Phase.DECODE,
        tokens=tokens,
        tp_degree=tp_degree
    )


if __name__ == "__main__":
    # Example usage
    print("LLM Task Examples:")
    print("=" * 70)

    # Prefill task
    prefill_task = create_prefill_task(
        task_id="T1",
        model_name="Qwen3",
        memory=30.0,
        weight=1.5,
        arrival_time=0.0,
        tokens=2048,
        tp_degree=2
    )
    print(f"\nPrefill Task:")
    print(f"  {prefill_task}")
    print(f"  Prefill tokens: {prefill_task.prefill_tokens}")
    print(f"  Decode tokens: {prefill_task.decode_tokens}")

    # Decode task
    decode_task = create_decode_task(
        task_id="T2",
        model_name="Qwen3",
        memory=30.0,
        weight=1.0,
        arrival_time=0.5,
        tokens=512,
        tp_degree=2
    )
    print(f"\nDecode Task:")
    print(f"  {decode_task}")
    print(f"  Prefill tokens: {decode_task.prefill_tokens}")
    print(f"  Decode tokens: {decode_task.decode_tokens}")

    # Simulate scheduling
    prefill_task.start_time = 0.0
    prefill_task.assigned_gpu_group = ["GPU-1", "GPU-2"]
    print(f"\nAfter scheduling:")
    print(f"  {prefill_task}")
    print(f"  Waiting time: {prefill_task.get_waiting_time():.2f}s")
