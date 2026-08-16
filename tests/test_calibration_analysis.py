import pytest

from resonance_asri.calibration.analysis import (
    aggregate_economics,
    classify_population,
    paired_deltas,
    summarize_by_class,
    summarize_populations,
)


def record(
    task_id: str,
    task_class: str,
    quality: float,
    input_tokens: int,
    output_tokens: int,
    calls: int,
    latency_ms: float,
) -> dict:
    return {
        "task_id": task_id,
        "task_class": task_class,
        "difficulty": "easy",
        "quality": quality,
        "answer": "a",
        "expected": "x",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "provider_calls": calls,
            "latency_ms": latency_ms,
        },
    }


def shallow_deep_pair(task_id: str, q_shallow: float, q_deep: float):
    shallow = record(task_id, "arithmetic", q_shallow, 100, 30, 1, 3500.0)
    deep = record(task_id, "arithmetic", q_deep, 700, 210, 7, 21000.0)
    return shallow, deep


def test_paired_deltas_compute_quality_and_cost_differences() -> None:
    shallow, deep = shallow_deep_pair("t1", 0.0, 1.0)
    rows = paired_deltas([shallow], [deep])

    assert len(rows) == 1
    row = rows[0]
    assert row["delta_quality"] == 1.0
    assert row["delta_tokens"] == (700 + 210) - (100 + 30)
    assert row["delta_calls"] == 6
    assert row["delta_latency_seconds"] == pytest.approx(21.0 - 3.5)
    assert row["c_tokens_deep"] == 910
    assert row["c_latency_deep_seconds"] == pytest.approx(21.0)


def test_paired_deltas_rejects_mismatched_task_sets() -> None:
    shallow, deep = shallow_deep_pair("t1", 0.0, 1.0)
    other, _ = shallow_deep_pair("t2", 0.0, 0.0)
    with pytest.raises(ValueError):
        paired_deltas([shallow, other], [deep])


def test_population_classification_four_mutually_exclusive_categories() -> None:
    # Partition over (shallow solved, deep solved).
    assert classify_population({"q_shallow": 1.0, "q_deep": 1.0}) == "stable_shallow_success"
    assert classify_population({"q_shallow": 0.0, "q_deep": 1.0}) == "compute_helps"
    assert classify_population({"q_shallow": 1.0, "q_deep": 0.0}) == "compute_hurts"
    assert classify_population({"q_shallow": 0.0, "q_deep": 0.0}) == "compute_does_nothing"
    # Partial (graded) scores count as unsolved; no bucket depends on delta_quality.
    assert classify_population({"q_shallow": 2 / 3, "q_deep": 1.0}) == "compute_helps"
    assert classify_population({"q_shallow": 1.0, "q_deep": 2 / 3}) == "compute_hurts"


def test_population_summary_counts_members_and_probabilities() -> None:
    rows = [
        {"task_id": "a", "task_class": "c", "q_shallow": 0.0, "q_deep": 1.0,
         "delta_quality": 1.0, "delta_tokens": 800, "delta_latency_seconds": 17.0,
         "c_latency_deep_seconds": 20.0},
        {"task_id": "b", "task_class": "c", "q_shallow": 0.0, "q_deep": 0.0,
         "delta_quality": 0.0, "delta_tokens": 800, "delta_latency_seconds": 17.0,
         "c_latency_deep_seconds": 20.0},
        {"task_id": "c", "task_class": "c", "q_shallow": 1.0, "q_deep": 1.0,
         "delta_quality": 0.0, "delta_tokens": 800, "delta_latency_seconds": 17.0,
         "c_latency_deep_seconds": 20.0},
        {"task_id": "d", "task_class": "c", "q_shallow": 1.0, "q_deep": 0.0,
         "delta_quality": -1.0, "delta_tokens": 800, "delta_latency_seconds": 17.0,
         "c_latency_deep_seconds": 20.0},
    ]
    summary = summarize_populations(rows)
    assert summary["counts"] == {
        "stable_shallow_success": 1,
        "compute_helps": 1,
        "compute_hurts": 1,
        "compute_does_nothing": 1,
    }
    assert summary["members"]["compute_hurts"] == ["d"]
    assert summary["probabilities"] == {
        "p_help": 0.25,
        "p_harm": 0.25,
        "p_no_value": 0.50,
    }
    assert summary["taxonomy"] == "mutually-exclusive-v2"


def test_by_class_aggregates_means() -> None:
    rows = [
        {"task_id": "a", "task_class": "math", "q_shallow": 0.0, "q_deep": 1.0,
         "delta_quality": 1.0, "delta_tokens": 800, "delta_latency_seconds": 17.0,
         "c_latency_deep_seconds": 21.0},
        {"task_id": "b", "task_class": "math", "q_shallow": 1.0, "q_deep": 1.0,
         "delta_quality": 0.0, "delta_tokens": 700, "delta_latency_seconds": 15.0,
         "c_latency_deep_seconds": 19.0},
    ]
    summaries = summarize_by_class(rows)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["tasks"] == 2
    assert summary["mean_q_shallow"] == 0.5
    assert summary["mean_q_deep"] == 1.0
    assert summary["mean_delta_quality"] == 0.5
    assert summary["mean_delta_tokens"] == 750.0
    assert summary["mean_c_latency_deep_seconds"] == 20.0


def test_aggregate_economics_quality_per_extra_token() -> None:
    rows = [
        {"q_shallow": 0.0, "q_deep": 1.0, "c_tokens_shallow": 66, "c_tokens_deep": 631,
         "c_latency_shallow_seconds": 0.6, "c_latency_deep_seconds": 9.8},
        {"q_shallow": 1.0, "q_deep": 1.0, "c_tokens_shallow": 66, "c_tokens_deep": 631,
         "c_latency_shallow_seconds": 0.6, "c_latency_deep_seconds": 9.8},
    ]
    economics = aggregate_economics(rows)
    assert economics["mean_q_shallow"] == 0.5
    assert economics["mean_q_deep"] == 1.0
    assert economics["delta_quality"] == 0.5
    assert economics["token_ratio"] == pytest.approx(631 / 66)
    assert economics["latency_ratio"] == pytest.approx(9.8 / 0.6)
    assert economics["quality_per_extra_token_under_uniform_deep"] == pytest.approx(
        0.5 / (631 - 66)
    )
