from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol, Sequence

from resonance_asri.contracts import ExecutionRequest


@dataclass(frozen=True, slots=True)
class ComputeDecision:
    """Observable compute allocation for one request."""

    reasoning_iterations: int = 1
    specialists: tuple[str, ...] = ()
    use_memory: bool = False
    verify: bool = False

    def __post_init__(self) -> None:
        if self.reasoning_iterations < 1:
            raise ValueError("reasoning_iterations must be at least 1")

    @property
    def nominal_units(self) -> int:
        """Simple provider-call proxy used before hardware FLOP telemetry exists."""

        # Each specialist adds one critique and one synthesis call.
        return (
            self.reasoning_iterations
            + (2 * len(self.specialists))
            + int(self.verify)
        )


class ComputePolicy(Protocol):
    policy_id: str

    def decide(self, request: ExecutionRequest) -> ComputeDecision: ...


class FixedShallowPolicy:
    policy_id = "fixed-shallow-v0"

    def decide(self, request: ExecutionRequest) -> ComputeDecision:
        del request
        return ComputeDecision(reasoning_iterations=1)


class FixedDeepPolicy:
    policy_id = "fixed-deep-v0"

    def decide(self, request: ExecutionRequest) -> ComputeDecision:
        del request
        return ComputeDecision(
            reasoning_iterations=4,
            specialists=("general-reviewer",),
            use_memory=False,
            verify=True,
        )


class AdaptiveHeuristicPolicy:
    """Transparent S0 policy used to validate the experiment harness.

    This is not intended as the final ASRI controller. It deliberately uses
    inspectable request-level features so routing behavior can be audited before
    any learned controller is introduced.
    """

    policy_id = "adaptive-heuristic-v0"

    _difficult_task_types = frozenset({"math", "code", "planning", "research"})
    _reasoning_markers = (
        "analyze",
        "compare",
        "debug",
        "design",
        "explain why",
        "prove",
        "reason",
        "tradeoff",
    )

    def decide(self, request: ExecutionRequest) -> ComputeDecision:
        text = request.prompt.casefold()
        score = 0

        if len(request.prompt) >= 320:
            score += 1
        if request.task_type in self._difficult_task_types:
            score += 1
        if any(marker in text for marker in self._reasoning_markers):
            score += 1
        if "\n" in request.prompt:
            score += 1

        if score == 0:
            return ComputeDecision(reasoning_iterations=1)
        if score == 1:
            return ComputeDecision(reasoning_iterations=2)

        specialist = request.task_type or "general-reviewer"
        if score == 2:
            return ComputeDecision(
                reasoning_iterations=3,
                specialists=(specialist,),
            )

        return ComputeDecision(
            reasoning_iterations=4,
            specialists=(specialist,),
            use_memory=request.task_type in {"planning", "research"},
            verify=True,
        )


class MatchedRandomPolicy:
    """Request-independent control sampling an empirical adaptive decision pool.

    Supplying the adaptive policy's observed decisions makes the marginal
    allocation distribution match in expectation while breaking the association
    between request difficulty and compute allocation.
    """

    policy_id = "matched-random-v0"

    def __init__(self, decision_pool: Sequence[ComputeDecision], seed: int) -> None:
        if not decision_pool:
            raise ValueError("decision_pool must not be empty")
        self._decision_pool = tuple(decision_pool)
        self._rng = Random(seed)

    def decide(self, request: ExecutionRequest) -> ComputeDecision:
        del request
        return self._rng.choice(self._decision_pool)
