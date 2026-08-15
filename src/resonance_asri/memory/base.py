from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from resonance_asri.contracts import ExecutionRequest


class MemoryProvider(Protocol):
    """Retrieval boundary used by conditional-memory experiments."""

    def retrieve(self, request: ExecutionRequest, *, limit: int = 4) -> Sequence[str]: ...
