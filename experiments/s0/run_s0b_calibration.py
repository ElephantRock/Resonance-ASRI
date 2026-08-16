#!/usr/bin/env python3
"""Run S0-B calibration conditions B0 (fixed-shallow) and B1 (fixed-deep).

Loads the frozen Qwen3-4B substrate once, emits the frozen task manifest,
runs every calibration task through the standard ASRIRuntime under both
fixed policies, scores responses with the deterministic evaluators, and
writes receipts + per-task records + paired-delta analysis.

Cost accounting (frozen for S0-B): C_tokens = input+output tokens, C_calls =
provider calls, C_latency = summed wall-clock seconds. Peak CUDA residency is
checked after every task as an abort guard (WDDM spills rather than raising
OOM on this node), never as an optimization target.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from resonance_asri.calibration import (
    build_manifest,
    classify_population,
    paired_deltas,
    score_response,
    summarize_by_class,
    summarize_populations,
)
from resonance_asri.calibration.tasks import CALIBRATION_TASKS
from resonance_asri.contracts import ExecutionRequest
from resonance_asri.controller import FixedDeepPolicy, FixedShallowPolicy
from resonance_asri.providers import QwenLocalProvider
from resonance_asri.reference.environment import capture_environment, write_environment_json
from resonance_asri.reference.loader import load_frozen_qwen_reference
from resonance_asri.runtime import ASRIRuntime
from resonance_asri.telemetry import JsonlReceiptLedger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "s0" / "artifacts" / "s0b"

CONDITIONS = (
    ("b0-fixed-shallow", FixedShallowPolicy),
    ("b1-fixed-deep", FixedDeepPolicy),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[s0b] output={out_dir}", flush=True)
    manifest = build_manifest()
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[s0b] frozen manifest: {manifest['task_count']} tasks "
        f"sha256={manifest['tasks_sha256'][:12]}",
        flush=True,
    )

    environment = capture_environment(REPO_ROOT)
    write_environment_json(out_dir / "environment.json", environment)

    loaded = load_frozen_qwen_reference()
    provider = QwenLocalProvider.from_reference(loaded, max_new_tokens=256)

    import torch  # runner-only; experiments scripts are never imported by CI

    total_vram = int(environment["gpu"]["total_vram_bytes"])
    resident_budget = total_vram - 2 * 1024**3  # desktop reserve, Phase-0B convention

    receipt_ledger = JsonlReceiptLedger(out_dir / "receipts.jsonl")
    task_records: dict[str, list[dict[str, Any]]] = {}

    for condition_name, policy_cls in CONDITIONS:
        runtime = ASRIRuntime(provider=provider, policy=policy_cls())
        records: list[dict[str, Any]] = []
        for index, task in enumerate(CALIBRATION_TASKS, start=1):
            started = datetime.now(UTC)
            request = ExecutionRequest(
                request_id=task.task_id, prompt=task.prompt, task_type=task.task_class
            )
            receipt = runtime.run(request)
            quality = score_response(receipt.answer, task)
            record = {
                "experiment_id": "s0b-calibration-v1",
                "condition": condition_name,
                "policy_id": receipt.policy_id,
                "task_id": task.task_id,
                "task_class": task.task_class,
                "difficulty": task.difficulty,
                "evaluator": task.evaluator,
                "expected": task.expected,
                "manifest_sha256": manifest["tasks_sha256"],
                "quality": quality,
                "answer": receipt.answer,
                "usage": asdict(receipt.usage),
                "timestamp_utc": started.isoformat(timespec="milliseconds"),
            }
            records.append(record)
            receipt_ledger.append(receipt)

            # Residency abort guard (safety metric only).
            peak = int(torch.cuda.max_memory_allocated())
            if peak > resident_budget:
                print(
                    f"[s0b] ABORT: peak_allocated={peak} > budget={resident_budget} "
                    f"at {condition_name}/{task.task_id}; WDDM spill regime",
                    flush=True,
                )
                _write_records(out_dir, task_records, records, condition_name)
                return 1

            latency_s = record["usage"]["latency_ms"] / 1000.0
            print(
                f"[s0b] {condition_name} {index:>2}/{len(CALIBRATION_TASKS)} "
                f"{task.task_id}: Q={quality:.1f} calls={record['usage']['provider_calls']} "
                f"tokens={record['usage']['input_tokens'] + record['usage']['output_tokens']} "
                f"latency={latency_s:.1f}s",
                flush=True,
            )
        task_records[condition_name] = records

    # Paired-delta analysis.
    rows = paired_deltas(task_records["b0-fixed-shallow"], task_records["b1-fixed-deep"])
    for row in rows:
        row["population"] = classify_population(row)
    populations = summarize_populations(rows)
    by_class = summarize_by_class(rows)

    analysis = {
        "experiment_id": "s0b-calibration-v1",
        "manifest_sha256": manifest["tasks_sha256"],
        "conditions": [name for name, _ in CONDITIONS],
        "task_count": len(rows),
        "populations": populations,
        "by_class": by_class,
        "per_task": rows,
    }
    (out_dir / "analysis.json").write_text(
        json.dumps(analysis, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"[s0b] populations: {json.dumps(populations['counts'], sort_keys=True)}",
        flush=True,
    )
    for summary in by_class:
        print(
            f"[s0b] class {summary['task_class']}: "
            f"q_shallow={summary['mean_q_shallow']:.2f} "
            f"q_deep={summary['mean_q_deep']:.2f} "
            f"d_tokens={summary['mean_delta_tokens']:.0f} "
            f"d_latency={summary['mean_delta_latency_seconds']:.1f}s",
            flush=True,
    )
    return 0


def _write_records(
    out_dir: Path,
    task_records: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    condition_name: str,
) -> None:
    all_records = [record for value in task_records.values() for record in value] + records
    (out_dir / "records_partial.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in all_records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
