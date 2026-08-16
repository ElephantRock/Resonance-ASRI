from resonance_asri.calibration.router import (
    HARM,
    HELP,
    NEUTRAL,
    build_matched_random_pool,
    confusion_matrix,
    controller_features,
    decision_is_consistent,
    decision_signature,
    routed_deep,
    three_state_crosstab,
    value_label,
)
from resonance_asri.contracts import ExecutionRequest
from resonance_asri.controller.policy import AdaptiveHeuristicPolicy, ComputeDecision


def decide(prompt: str, task_type: str | None = None) -> ComputeDecision:
    request = ExecutionRequest(request_id="r", prompt=prompt, task_type=task_type)
    return AdaptiveHeuristicPolicy().decide(request)


def test_features_match_real_policy_decisions() -> None:
    dice_prompt = (
        "When two fair six-sided dice are rolled, what is the probability that "
        "the two numbers sum to 7? Answer with only the fraction, like 1/2."
    )
    cases = [
        ("What is 2 + 2?", None),  # short, plain: score 0, shallow
        (dice_prompt, None),
        ("What does this code print?\n\nprint(len('engines'))", None),  # multiline
        ("Analyze the tradeoff between two designs.", "math"),  # marker + difficult type
        ("word " * 100, None),  # length >= 320
        ("multi\nline and " + "long " * 100, "research"),  # all four features
    ]
    for prompt, task_type in cases:
        features = controller_features(prompt, task_type)
        decision = decide(prompt, task_type)
        assert decision_is_consistent(features, decision), (prompt[:40], features)


def test_feature_accounting_is_explicit() -> None:
    features = controller_features("Analyze the tradeoff.\nSecond line.", "math")
    assert features["length_ge_320"] is False
    assert features["task_type_difficult"] is True
    assert features["reasoning_markers_hit"] == ["analyze", "tradeoff"]
    assert features["multiline"] is True
    assert features["score"] == 3


def test_routed_deep_captures_any_extra_compute() -> None:
    assert routed_deep(ComputeDecision(reasoning_iterations=1)) is False
    assert routed_deep(ComputeDecision(reasoning_iterations=2)) is True
    assert routed_deep(ComputeDecision(reasoning_iterations=1, verify=True)) is True
    assert routed_deep(
        ComputeDecision(reasoning_iterations=1, specialists=("s",))
    ) is True


def test_value_label_three_states() -> None:
    assert value_label(0.0, 1.0) == HELP
    assert value_label(1.0, 0.0) == HARM
    assert value_label(1.0, 1.0) == NEUTRAL
    assert value_label(0.0, 0.0) == NEUTRAL
    assert value_label(2 / 3, 1.0) == HELP  # partial counts as unsolved


def record(task_id: str, deep: bool, label: str) -> dict:
    return {"task_id": task_id, "routed_deep": deep, "value_label": label}


def test_confusion_matrix_counts() -> None:
    records = [
        record("tp", True, HELP),
        record("fp", True, NEUTRAL),
        record("fp2", True, HARM),
        record("fn", False, HELP),
        record("tn", False, NEUTRAL),
        record("tn2", False, HARM),
    ]
    assert confusion_matrix(records) == {
        "deep_routed_help": 1,
        "deep_routed_no_help": 2,
        "shallow_routed_help": 1,
        "shallow_routed_no_help": 2,
    }


def test_three_state_crosstab_preserves_harm_row() -> None:
    records = [
        record("a", True, NEUTRAL),
        record("b", False, HARM),
        record("c", True, HARM),
    ]
    table = three_state_crosstab(records)
    assert table["deep_routed"][HARM] == 1
    assert table["shallow_routed"][HARM] == 1
    assert table["deep_routed"][NEUTRAL] == 1


def test_pool_builder_counts_distribution_and_hash() -> None:
    decisions = [
        ComputeDecision(reasoning_iterations=1),
        ComputeDecision(reasoning_iterations=1),
        ComputeDecision(reasoning_iterations=2),
        ComputeDecision(reasoning_iterations=4, specialists=("code",), verify=True),
    ]
    pool = build_matched_random_pool(decisions, seed=20260816, source="test")
    assert pool["pool_size"] == 4
    assert pool["distribution"] == {
        "iter=1;specialists=;memory=0;verify=0": 2,
        "iter=2;specialists=;memory=0;verify=0": 1,
        "iter=4;specialists=code;memory=0;verify=1": 1,
    }
    assert len(pool["pool_sha256"]) == 64

    again = build_matched_random_pool(decisions, seed=20260816, source="test")
    assert again["pool_sha256"] == pool["pool_sha256"]

    reordered = [decisions[2], decisions[1], decisions[0], decisions[3]]
    reordered_pool = build_matched_random_pool(reordered, seed=20260816, source="test")
    # Identical multiset, different order: the hash pins the exact observed
    # sequence, so pool composition and order are both auditable.
    assert reordered_pool["distribution"] == pool["distribution"]
    assert reordered_pool["pool_sha256"] != pool["pool_sha256"]


def test_decision_signature_is_canonical() -> None:
    assert decision_signature(ComputeDecision(reasoning_iterations=1)) == (
        "iter=1;specialists=;memory=0;verify=0"
    )
