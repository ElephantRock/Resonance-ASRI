"""Router characterization for the frozen adaptive heuristic.

Step-6 tooling: recompute the controller's transparent features and score from
the frozen policy's own constants, label each task with its empirical
value-of-compute state (HELP / NEUTRAL / HARM), build routing confusion
matrices, and derive the matched-random decision pool. Everything here is pure
and CPU-testable; the frozen policy code itself is never modified.

Provenance rule (see S0B_ROUTER_RESULT.md): adaptive-heuristic-v0 predates the
S0-B shallow/deep calibration and was not tuned on its outcomes. Feature
recomputation mirrors AdaptiveHeuristicPolicy.decide and is checked against the
policy's real decisions at run time — a mismatch fails loudly rather than
silently drifting.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from resonance_asri.controller.policy import AdaptiveHeuristicPolicy, ComputeDecision

HELP = "HELP"
NEUTRAL = "NEUTRAL"
HARM = "HARM"


def controller_features(prompt: str, task_type: str | None) -> dict[str, Any]:
    """Mirror AdaptiveHeuristicPolicy's transparent scoring, feature by feature."""

    policy = AdaptiveHeuristicPolicy()
    text = prompt.casefold()
    length_ge_320 = len(prompt) >= 320
    task_type_difficult = task_type in policy._difficult_task_types
    markers_hit = [marker for marker in policy._reasoning_markers if marker in text]
    multiline = "\n" in prompt
    score = int(length_ge_320) + int(task_type_difficult) + int(bool(markers_hit)) + int(
        multiline
    )
    return {
        "length_ge_320": length_ge_320,
        "task_type_difficult": task_type_difficult,
        "reasoning_markers_hit": markers_hit,
        "multiline": multiline,
        "score": score,
    }


def decision_is_consistent(features: dict[str, Any], decision: ComputeDecision) -> bool:
    """True when the recomputed score explains the observed decision.

    Mapping mirrored from AdaptiveHeuristicPolicy.decide: score 0 -> 1
    iteration, 1 -> 2, 2 -> 3 + specialist, 3+ -> 4 + specialist (+ memory for
    planning/research task types) + verification.
    """

    score = features["score"]
    expected_iterations = {0: 1, 1: 2, 2: 3}.get(score, 4)
    expected_specialists = score >= 2
    expected_verify = score >= 3
    return (
        decision.reasoning_iterations == expected_iterations
        and bool(decision.specialists) == expected_specialists
        and decision.verify == expected_verify
    )


def routed_deep(decision: ComputeDecision) -> bool:
    """Any allocation beyond a single answer pass counts as deep routing."""

    return decision.reasoning_iterations > 1 or bool(decision.specialists) or decision.verify


def decision_signature(decision: ComputeDecision) -> str:
    return (
        f"iter={decision.reasoning_iterations};"
        f"specialists={','.join(decision.specialists)};"
        f"memory={int(decision.use_memory)};"
        f"verify={int(decision.verify)}"
    )


def value_label(q_shallow: float, q_deep: float) -> str:
    """Three-state empirical value of extra compute for one task."""

    shallow_solved = q_shallow >= 1.0
    deep_solved = q_deep >= 1.0
    if not shallow_solved and deep_solved:
        return HELP
    if shallow_solved and not deep_solved:
        return HARM
    return NEUTRAL


def confusion_matrix(records: list[dict[str, Any]]) -> dict[str, int]:
    """Routing vs actual usefulness of extra compute (2x2 counts)."""

    matrix = {
        "deep_routed_help": 0,
        "deep_routed_no_help": 0,
        "shallow_routed_help": 0,
        "shallow_routed_no_help": 0,
    }
    for record in records:
        deep = record["routed_deep"]
        help_needed = record["value_label"] == HELP
        if deep and help_needed:
            matrix["deep_routed_help"] += 1
        elif deep:
            matrix["deep_routed_no_help"] += 1
        elif help_needed:
            matrix["shallow_routed_help"] += 1
        else:
            matrix["shallow_routed_no_help"] += 1
    return matrix


def three_state_crosstab(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Routing (deep/shallow) x value state (HELP/NEUTRAL/HARM) counts."""

    table = {
        "deep_routed": {HELP: 0, NEUTRAL: 0, HARM: 0},
        "shallow_routed": {HELP: 0, NEUTRAL: 0, HARM: 0},
    }
    for record in records:
        row = "deep_routed" if record["routed_deep"] else "shallow_routed"
        table[row][record["value_label"]] += 1
    return table


def decision_as_pool_entry(decision: ComputeDecision) -> dict[str, Any]:
    return {
        "reasoning_iterations": decision.reasoning_iterations,
        "specialists": list(decision.specialists),
        "use_memory": decision.use_memory,
        "verify": decision.verify,
    }


def build_matched_random_pool(
    decisions: list[ComputeDecision],
    *,
    seed: int,
    source: str,
) -> dict[str, Any]:
    """Persistable empirical allocation pool for matched-random-v0.

    The pool is the exact multiset of the frozen controller's observed
    decisions. For S0-C, prefer exact allocation-count matching: shuffle this
    pool as a fixed-size vector (sample without replacement) so adaptive and
    random spend the same logical allocation counts by construction; actual
    token accounting remains the primary cost measure because prompt and
    refinement lengths differ per task.
    """

    pool = [decision_as_pool_entry(decision) for decision in decisions]
    distribution: dict[str, int] = {}
    for decision in decisions:
        signature = decision_signature(decision)
        distribution[signature] = distribution.get(signature, 0) + 1
    digest = hashlib.sha256(
        json.dumps(pool, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "source": source,
        "pool_size": len(pool),
        "sampling_seed": seed,
        "pool": pool,
        "distribution": dict(sorted(distribution.items())),
        "pool_sha256": digest,
        "sampling_rule": "exact allocation-count matching preferred for S0-C: "
        "shuffle this pool without replacement; token accounting remains primary",
    }
