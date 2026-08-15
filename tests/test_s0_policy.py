from resonance_asri.contracts import ExecutionRequest
from resonance_asri.controller import (
    AdaptiveHeuristicPolicy,
    ComputeDecision,
    FixedDeepPolicy,
    FixedShallowPolicy,
    MatchedRandomPolicy,
)


def test_fixed_policy_compute_units_are_explicit() -> None:
    request = ExecutionRequest(request_id="r1", prompt="hello")

    shallow = FixedShallowPolicy().decide(request)
    deep = FixedDeepPolicy().decide(request)

    assert shallow.nominal_units == 1
    assert deep.nominal_units == 7


def test_adaptive_policy_keeps_easy_request_shallow() -> None:
    request = ExecutionRequest(request_id="r1", prompt="What is 2 + 2?")
    decision = AdaptiveHeuristicPolicy().decide(request)

    assert decision == ComputeDecision(reasoning_iterations=1)


def test_adaptive_policy_escalates_difficult_research_request() -> None:
    request = ExecutionRequest(
        request_id="r2",
        task_type="research",
        prompt=("Analyze the tradeoff between two designs.\n" + ("evidence " * 50)),
    )
    decision = AdaptiveHeuristicPolicy().decide(request)

    assert decision.reasoning_iterations == 4
    assert decision.specialists == ("research",)
    assert decision.use_memory is True
    assert decision.verify is True


def test_matched_random_policy_is_request_independent_and_seeded() -> None:
    pool = [
        ComputeDecision(reasoning_iterations=1),
        ComputeDecision(reasoning_iterations=4, verify=True),
    ]
    left = MatchedRandomPolicy(pool, seed=17)
    right = MatchedRandomPolicy(pool, seed=17)

    requests = [
        ExecutionRequest(request_id="a", prompt="easy"),
        ExecutionRequest(request_id="b", prompt="hard and complicated"),
        ExecutionRequest(request_id="c", prompt="different again"),
    ]

    assert [left.decide(item) for item in requests] == [
        right.decide(item) for item in reversed(requests)
    ]
