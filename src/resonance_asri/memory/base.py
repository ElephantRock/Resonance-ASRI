from __future__ import annotations

from typing import Protocol, Sequence

from resonance_asri.contracts import ExecutionRequest


class MemoryProvider(Protocol):
    """Retrieval boundary used by conditional-memory experiments."""

    def retrieve(self, request: ExecutionRequest, *, limit: int = 4) -> Sequence[str]: ...
