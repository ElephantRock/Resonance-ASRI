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
    """Assign a task to one of the four compute-value populations."""

    if row["q_shallow"] >= 1.0:
        # Shallow already solves it; extra compute can only waste or damage.
        return "shallow_already_solves"
    if row["delta_quality"] > 0:
        return "compute_helps"
    if row["delta_quality"] < 0:
        return "compute_hurts"
    return "compute_does_nothing"


def summarize_populations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    populations = {
        "compute_helps": [],
        "compute_does_nothing": [],
        "compute_hurts": [],
        "shallow_already_solves": [],
    }
    for row in rows:
        populations[classify_population(row)].append(row["task_id"])

    return {
        "counts": {name: len(members) for name, members in populations.items()},
        "members": populations,
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


def selective_value_exists(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Does extra computation have selective (class- or task-level) value?

    A gradient exists when extra compute helps somewhere (any task, or any
    class on average) while doing nothing or hurting somewhere else. If deep
    uniformly dominates everywhere, the set cannot demonstrate *selective*
    allocation value; if deep never helps, adaptive routing has nothing to
    buy. Both facts matter for freezing the adaptive policy.
    """

    helps_any_task = any(row["delta_quality"] > 0 for row in summaries)
    class_means = [item["mean_delta_quality"] for item in summaries]
    helps_any_class = any(mean > 0 for mean in class_means)
    flat_or_negative_class = any(mean <= 0 for mean in class_means)
    return {
        "helps_any_task": helps_any_task,
        "helps_any_class": helps_any_class,
        "mixed_class_gradient": helps_any_class and flat_or_negative_class,
        "note": "mixed_class_gradient=True means adaptive allocation has class-level "
        "signal to exploit; task-level mixed signal additionally requires the "
        "population table",
    }
