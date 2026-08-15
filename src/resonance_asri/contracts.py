from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Provider-agnostic request entering the ASRI runtime."""

    request_id: str
    prompt: str
    task_type: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Resource measurements for one completed ASRI execution."""

    input_tokens: int = 0
    output_tokens: int = 0
    provider_calls: int = 0
    reasoning_iterations: int = 0
    retrieval_count: int = 0
    verifier_count: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float | None = None

    def __post_init__(self) -> None:
        nonnegative = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "provider_calls": self.provider_calls,
            "reasoning_iterations": self.reasoning_iterations,
            "retrieval_count": self.retrieval_count,
            "verifier_count": self.verifier_count,
            "tool_calls": self.tool_calls,
            "latency_ms": self.latency_ms,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.estimated_cost_usd is not None and self.estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd must be non-negative")


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """World-safe evidence emitted by the ASRI runtime.

    The receipt records control decisions and resource use without requiring
    private chain-of-thought or hidden model activations.
    """

    run_id: str
    request_id: str
    policy_id: str
    model_id: str
    answer: str
    usage: ResourceUsage
    specialists_activated: Sequence[str] = ()
    outcome_metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)
