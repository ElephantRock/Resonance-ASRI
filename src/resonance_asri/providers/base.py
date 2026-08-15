from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Minimal completion response with accounting data."""

    text: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        if self.estimated_cost_usd is not None and self.estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd must be non-negative")


class CompletionProvider(Protocol):
    """Model-provider boundary for S0.

    Implementations may call a hosted API or a local model. The runtime depends
    only on this interface and never on provider-specific SDK objects.
    """

    @property
    def model_id(self) -> str: ...

    def complete(self, *, prompt: str, purpose: str) -> ProviderResponse: ...
