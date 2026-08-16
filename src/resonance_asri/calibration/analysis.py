"""Per-task paired-delta analysis for S0-B calibration.

All logic here is pure (CPU-testable): the runner feeds it per-task records
carrying quality and cost for the shallow and deep conditions, and gets back
per-task deltas, the four behavior populations, and per-class aggregates.

Cost accounting (frozen for S0-B):
- C_tokens  = total input + output tokens across all calls of the run
- C_calls   = number of model calls
- C_latency = summed wall-clock latency (seconds)
Peak residency is tracked as a safety metric only.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any


def task_cost_tokens(record: dict[str, Any]) -> int:
    usage = record["usage"]
    return int(usage["input_tokens"]) + int(usage["output_tokens"])


def task_cost_calls(record: dict[str, Any]) -> int:
    return int(record["usage"]["provider_calls"])


def task_cost_latency(record: dict[str, Any]) -> float:
    return float(record["usage"]["latency_ms"]) / 1000.0


def paired_deltas(
    shallow_records: list[dict[str, Any]],
    deep_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join shallow/deep records by task_id and compute per-task deltas."""

    shallow_by_id = {record["task_id"]: record for record in shallow_records}
    deep_by_id = {record["task_id"]: record for record in deep_records}
    if set(shallow_by_id) != set(deep_by_id):
        missing_in_deep = sorted(set(shallow_by_id) - set(deep_by_id))
        missing_in_shallow = sorted(set(deep_by_id) - set(shallow_by_id))
        raise ValueError(
            f"task sets differ: missing_in_deep={missing_in_deep} "
            f"missing_in_shallow={missing_in_shallow}"
        )

    rows: list[dict[str, Any]] = []
    for task_id in sorted(shallow_by_id):
        shallow = shallow_by_id[task_id]
        deep = deep_by_id[task_id]
        rows.append(
            {
                "task_id": task_id,
                "task_class": shallow["task_class"],
                "difficulty": shallow["difficulty"],
                "q_shallow": shallow["quality"],
                "q_deep": deep["quality"],
                "delta_quality": deep["quality"] - shallow["quality"],
                "delta_tokens": task_cost_tokens(deep) - task_cost_tokens(shallow),
                "delta_calls": task_cost_calls(deep) - task_cost_calls(shallow),
                "delta_latency_seconds": task_cost_latency(deep)
                - task_cost_latency(shallow),
                "c_tokens_shallow": task_cost_tokens(shallow),
                "c_tokens_deep": task_cost_tokens(deep),
                "c_calls_shallow": task_cost_calls(shallow),
                "c_calls_deep": task_cost_calls(deep),
                "c_latency_shallow_seconds": task_cost_latency(shallow),
                "c_latency_deep_seconds": task_cost_latency(deep),
                "answer_shallow": shallow["answer"],
                "answer_deep": deep["answer"],
                "expected": shallow["expected"],
            }
        )
    return rows


def classify_population(row: dict[str, Any]) -> str:
    """Assign a task to one of four mutually exclusive populations.

    Solved means quality >= 1.0. The partition is over the pair
    (shallow solved, deep solved), so every task lands in exactly one bucket:
      stable_shallow_success: shallow solved AND deep stayed solved
      compute_helps:          shallow failed AND deep solved
      compute_hurts:          shallow solved AND deep broke it
      compute_does_nothing:   both failed
    """

    shallow_solved = row["q_shallow"] >= 1.0
    deep_solved = row["q_deep"] >= 1.0
    if shallow_solved and deep_solved:
        return "stable_shallow_success"
    if deep_solved:
        return "compute_helps"
    if shallow_solved:
        return "compute_hurts"
    return "compute_does_nothing"


POPULATION_ORDER = (
    "stable_shallow_success",
    "compute_helps",
    "compute_hurts",
    "compute_does_nothing",
)


def summarize_populations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    populations = {name: [] for name in POPULATION_ORDER}
    for row in rows:
        populations[classify_population(row)].append(row["task_id"])

    total = len(rows)
    counts = {name: len(members) for name, members in populations.items()}
    no_value = counts["stable_shallow_success"] + counts["compute_does_nothing"]
    return {
        "taxonomy": "mutually-exclusive-v2",
        "counts": counts,
        "members": populations,
        "probabilities": {
            "p_help": counts["compute_helps"] / total if total else None,
            "p_harm": counts["compute_hurts"] / total if total else None,
            "p_no_value": no_value / total if total else None,
        },
    }


def aggregate_economics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate quality/cost economics, including quality per extra token."""

    total = len(rows)
    if not total:
        return {}
    mean_q_shallow = fmean(row["q_shallow"] for row in rows)
    mean_q_deep = fmean(row["q_deep"] for row in rows)
    mean_tokens_shallow = fmean(row["c_tokens_shallow"] for row in rows)
    mean_tokens_deep = fmean(row["c_tokens_deep"] for row in rows)
    mean_latency_shallow = fmean(row["c_latency_shallow_seconds"] for row in rows)
    mean_latency_deep = fmean(row["c_latency_deep_seconds"] for row in rows)
    delta_quality = mean_q_deep - mean_q_shallow
    delta_tokens = mean_tokens_deep - mean_tokens_shallow
    return {
        "mean_q_shallow": mean_q_shallow,
        "mean_q_deep": mean_q_deep,
        "delta_quality": delta_quality,
        "mean_c_tokens_shallow": mean_tokens_shallow,
        "mean_c_tokens_deep": mean_tokens_deep,
        "token_ratio": mean_tokens_deep / mean_tokens_shallow if mean_tokens_shallow else None,
        "mean_c_latency_shallow_seconds": mean_latency_shallow,
        "mean_c_latency_deep_seconds": mean_latency_deep,
        "latency_ratio": mean_latency_deep / mean_latency_shallow
        if mean_latency_shallow
        else None,
        "quality_per_extra_token_under_uniform_deep": (
            delta_quality / delta_tokens if delta_tokens else None
        ),
    }


def summarize_by_class(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_class"]].append(row)

    summaries = []
    for task_class, members in sorted(grouped.items()):
        summaries.append(
            {
                "task_class": task_class,
                "tasks": len(members),
                "mean_q_shallow": fmean(row["q_shallow"] for row in members),
                "mean_q_deep": fmean(row["q_deep"] for row in members),
                "mean_delta_quality": fmean(row["delta_quality"] for row in members),
                "mean_delta_tokens": fmean(row["delta_tokens"] for row in members),
                "mean_delta_latency_seconds": fmean(
                    row["delta_latency_seconds"] for row in members
                ),
                "mean_c_latency_deep_seconds": fmean(
                    row["c_latency_deep_seconds"] for row in members
                ),
            }
        )
    return summaries


def selective_value_exists(
    rows: list[dict[str, Any]],
    class_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Does extra computation have selective (class- or task-level) value?

    A gradient exists when extra compute helps somewhere (any task, or any
    class on average) while doing nothing or hurting somewhere else. If deep
    uniformly dominates everywhere, the set cannot demonstrate *selective*
    allocation value; if deep never helps, adaptive routing has nothing to
    buy. Both facts matter for freezing the adaptive policy.
    """

    helps_any_task = any(row["delta_quality"] > 0 for row in rows)
    harms_any_task = any(row["delta_quality"] < 0 for row in rows)
    class_means = [item["mean_delta_quality"] for item in class_summaries]
    helps_any_class = any(mean > 0 for mean in class_means)
    flat_or_negative_class = any(mean <= 0 for mean in class_means)
    return {
        "helps_any_task": helps_any_task,
        "harms_any_task": harms_any_task,
        "helps_any_class": helps_any_class,
        "mixed_class_gradient": helps_any_class and flat_or_negative_class,
        "note": "mixed_class_gradient=True means adaptive allocation has class-level "
        "signal to exploit; task-level mixed signal additionally requires the "
        "population table",
    }
