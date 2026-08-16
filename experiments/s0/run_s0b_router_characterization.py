#!/usr/bin/env python3
"""Characterize the frozen adaptive heuristic on the S0-B calibration tasks.

Step 6 of the S0-B sequence. This is router characterization data, not
evidence for the main ASRI claim. For every calibration task it records the
controller's features/score/decision, the executed adaptive quality and cost,
the empirical value label (HELP/NEUTRAL/HARM from the shallow/deep passes),
routing confusion matrices, the specific routing checks of interest, and the
matched-random decision pool derived from the controller's observed
allocations.

Provenance: adaptive-heuristic-v0 was created in scaffold commit e7eb10e and
last touched by the Ruff import fix 896df71, both before any S0-B calibration
measurement. It is frozen prospectively; its alignment with the observed
class gradient is an observation, not a tuning decision.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from resonance_asri.calibration import score_response
from resonance_asri.calibration.router import (
    build_matched_random_pool,
    confusion_matrix,
    controller_features,
    decision_is_consistent,
    decision_signature,
    routed_deep,
    three_state_crosstab,
    value_label,
)
from resonance_asri.calibration.tasks import CALIBRATION_TASKS
from resonance_asri.contracts import ExecutionRequest
from resonance_asri.controller import AdaptiveHeuristicPolicy
from resonance_asri.providers import QwenLocalProvider
from resonance_asri.reference.environment import capture_environment, write_environment_json
from resonance_asri.reference.loader import load_frozen_qwen_reference
from resonance_asri.runtime import ASRIRuntime
from resonance_asri.telemetry import JsonlReceiptLedger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "s0" / "artifacts" / "s0b_router"
CALIBRATION_ANALYSIS = REPO_ROOT / "research" / "s0" / "artifacts" / "s0b" / "analysis.json"
POOL_SAMPLING_SEED = 20260816

ROUTING_CHECK_TASK_IDS = {
    "compute_helps_routed_deep?": ("arith-04", "code-02", "code-03", "word-06"),
    "harm_cases_routed_shallow?": ("instr-03", "logic-06"),
    "spiral_case_logic-04": ("logic-04",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = json.loads(CALIBRATION_ANALYSIS.read_text(encoding="utf-8"))
    shallow_deep = {row["task_id"]: row for row in calibration["per_task"]}
    manifest_sha = calibration["manifest_sha256"]

    print(f"[router] output={out_dir}", flush=True)
    environment = capture_environment(REPO_ROOT)
    write_environment_json(out_dir / "environment.json", environment)

    loaded = load_frozen_qwen_reference()
    provider = QwenLocalProvider.from_reference(loaded=loaded, max_new_tokens=256)
    policy = AdaptiveHeuristicPolicy()
    runtime = ASRIRuntime(provider=provider, policy=policy)

    import torch  # runner-only import; never imported by CI

    resident_budget = int(environment["gpu"]["total_vram_bytes"]) - 2 * 1024**3

    ledger = JsonlReceiptLedger(out_dir / "receipts.jsonl")
    records: list[dict[str, Any]] = []
    decisions_by_task: dict[str, Any] = {}

    for index, task in enumerate(CALIBRATION_TASKS, start=1):
        request = ExecutionRequest(
            request_id=task.task_id, prompt=task.prompt, task_type=task.task_class
        )
        features = controller_features(task.prompt, task.task_class)
        decision = policy.decide(request)
        if not decision_is_consistent(features, decision):
            raise SystemExit(
                f"router feature drift on {task.task_id}: features={features} "
                f"decision={decision}"
            )
        decisions_by_task[task.task_id] = decision

        receipt = runtime.run(request)
        ledger.append(receipt)
        quality = score_response(receipt.answer, task)
        baseline = shallow_deep[task.task_id]
        usage = asdict(receipt.usage)
        record = {
            "experiment_id": "s0b-router-characterization-v1",
            "task_id": task.task_id,
            "task_class": task.task_class,
            "difficulty": task.difficulty,
            "manifest_sha256": manifest_sha,
            "shallow_quality": baseline["q_shallow"],
            "deep_quality": baseline["q_deep"],
            "delta_quality": baseline["delta_quality"],
            "value_label": value_label(baseline["q_shallow"], baseline["q_deep"]),
            "adaptive_decision": decision_signature(decision),
            "allocated_depth": decision.reasoning_iterations,
            "controller_score": features["score"],
            "controller_features": features,
            "routed_deep": routed_deep(decision),
            "adaptive_quality": quality,
            "adaptive_tokens": usage["input_tokens"] + usage["output_tokens"],
            "adaptive_calls": usage["provider_calls"],
            "adaptive_latency_seconds": usage["latency_ms"] / 1000.0,
            "answer_adaptive": receipt.answer,
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="milliseconds"),
        }
        records.append(record)

        peak = int(torch.cuda.max_memory_allocated())
        if peak > resident_budget:
            print(
                f"[router] ABORT: peak_allocated={peak} > budget={resident_budget} "
                f"at {task.task_id}",
                flush=True,
            )
            break

        print(
            f"[router] {index:>2}/{len(CALIBRATION_TASKS)} {task.task_id}: "
            f"label={record['value_label']:<7} routed={'deep' if record['routed_deep'] else 'shallow'} "
            f"depth={record['allocated_depth']} score={record['controller_score']} "
            f"Q={quality:.1f} tokens={record['adaptive_tokens']} "
            f"latency={record['adaptive_latency_seconds']:.1f}s",
            flush=True,
        )

    # Analysis: confusion matrices, routing checks, decision distribution.
    matrix = confusion_matrix(records)
    crosstab = three_state_crosstab(records)
    routing_checks: dict[str, Any] = {}
    for check_name, task_ids in ROUTING_CHECK_TASK_IDS.items():
        routing_checks[check_name] = [
            {
                "task_id": record["task_id"],
                "routed_deep": record["routed_deep"],
                "allocated_depth": record["allocated_depth"],
                "value_label": record["value_label"],
                "adaptive_tokens": record["adaptive_tokens"],
                "adaptive_quality": record["adaptive_quality"],
            }
            for record in records
            if record["task_id"] in task_ids
        ]

    distribution: dict[str, int] = {}
    for record in records:
        distribution[record["adaptive_decision"]] = distribution.get(
            record["adaptive_decision"], 0
        ) + 1

    pool = build_matched_random_pool(
        [decisions_by_task[record["task_id"]] for record in records],
        seed=POOL_SAMPLING_SEED,
        source="adaptive-heuristic-v0 on s0b-calibration-v1 (36 tasks)",
    )

    mean_quality = sum(r["adaptive_quality"] for r in records) / len(records)
    mean_tokens = sum(r["adaptive_tokens"] for r in records) / len(records)
    mean_latency = sum(r["adaptive_latency_seconds"] for r in records) / len(records)
    economics = {
        "mean_adaptive_quality": mean_quality,
        "mean_adaptive_tokens": mean_tokens,
        "mean_adaptive_calls": sum(r["adaptive_calls"] for r in records) / len(records),
        "mean_adaptive_latency_seconds": mean_latency,
        "reference_mean_shallow_quality": calibration["economics"]["mean_q_shallow"],
        "reference_mean_deep_quality": calibration["economics"]["mean_q_deep"],
        "reference_mean_shallow_tokens": calibration["economics"]["mean_c_tokens_shallow"],
        "reference_mean_deep_tokens": calibration["economics"]["mean_c_tokens_deep"],
    }

    analysis = {
        "experiment_id": "s0b-router-characterization-v1",
        "purpose": "router characterization; not evidence for the main ASRI claim",
        "provenance": {
            "policy": "adaptive-heuristic-v0",
            "created_in_commit": "e7eb10e (scaffold ASRI-S0 adaptive compute experiment)",
            "last_modified_in_commit": "896df71 (Ruff import fix only)",
            "statement": "existed before S0-B shallow/deep calibration, was not modified "
            "using S0-B outcomes, frozen prospectively for adaptive evaluation; apparent "
            "alignment with the observed class gradient is an observation, not tuning",
        },
        "manifest_sha256": manifest_sha,
        "task_count": len(records),
        "confusion_matrix": matrix,
        "three_state_crosstab": crosstab,
        "routing_checks": routing_checks,
        "decision_distribution": dict(sorted(distribution.items())),
        "economics": economics,
        "records": records,
    }
    (out_dir / "analysis.json").write_text(
        json.dumps(analysis, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "matched_random_pool.json").write_text(
        json.dumps(pool, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[router] confusion: {json.dumps(matrix, sort_keys=True)}", flush=True)
    print(f"[router] crosstab: {json.dumps(crosstab, sort_keys=True)}", flush=True)
    print(
        f"[router] distribution: {json.dumps(dict(sorted(distribution.items())), sort_keys=True)}",
        flush=True,
    )
    print(
        f"[router] adaptive: mean_Q={mean_quality:.3f} mean_tokens={mean_tokens:.0f} "
        f"mean_latency={mean_latency:.1f}s (shallow Q="
        f"{calibration['economics']['mean_q_shallow']:.3f}, deep Q="
        f"{calibration['economics']['mean_q_deep']:.3f})",
        flush=True,
    )
    print(
        f"[router] matched-random pool: {pool['pool_size']} decisions, seed="
        f"{POOL_SAMPLING_SEED}, sha256={pool['pool_sha256'][:16]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
