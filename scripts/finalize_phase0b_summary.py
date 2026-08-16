#!/usr/bin/env python3
"""Regenerate Phase-0B summary artifacts from raw runs.jsonl evidence.

Used after a terminated sweep: reads the raw measured records, separates
resident runs from WDDM shared-memory spill-regime runs (which are invalid as
in-VRAM envelope evidence), aggregates the resident runs, recomputes the
envelopes, and rewrites summary.csv/summary.json. runs.jsonl and
environment.json are never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from resonance_asri.reference import envelope

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "research" / "s0" / "artifacts" / "phase0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument(
        "--termination-reason",
        default=None,
        help="Why the sweep ended before completing the grid, if it did.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = args.artifacts_dir

    records = [
        json.loads(line)
        for line in (artifacts / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    environment = json.loads((artifacts / "environment.json").read_text(encoding="utf-8"))
    previous = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))

    total_vram = int(environment["gpu"]["total_vram_bytes"])
    resident, spill_regime = envelope.classify_residency(
        records, total_vram_bytes=total_vram
    )
    summaries = envelope.aggregate_runs(resident)
    envelopes = envelope.determine_envelopes(summaries, total_vram_bytes=total_vram)

    methodology: dict[str, Any] = dict(previous.get("methodology") or {})
    if args.termination_reason:
        methodology["terminated_early"] = True
        methodology["termination_reason"] = args.termination_reason
    methodology["invalidated_runs"] = [
        {
            "run_id": record["run_id"],
            "requested_prompt_tokens": record["requested_prompt_tokens"],
            "max_new_tokens": record["max_new_tokens"],
            "repetition": record["repetition"],
            "peak_allocated_bytes": record["peak_allocated_bytes"],
            "latency_seconds": record["latency_seconds"],
            "reason": "wddm_shared_memory_spill_regime",
        }
        for record in spill_regime
    ]
    methodology["residency_rule"] = {
        "resident_budget_bytes": total_vram - envelope.DESKTOP_VRAM_RESERVE_BYTES,
        "definition": "peak_allocated_bytes above total VRAM minus desktop reserve implies "
        "WDDM paging into shared system memory; such runs are excluded from aggregation",
    }

    payload = {
        "phase": "phase0b",
        "experiment_id": envelope.EXPERIMENT_ID,
        "reference": previous.get("reference"),
        "methodology": methodology,
        "conditions": summaries,
        "envelopes": envelopes,
        "environment_ref": "environment.json",
    }
    envelope.write_summary_csv(artifacts / "summary.csv", summaries)
    envelope.write_summary_json(artifacts / "summary.json", payload)

    print(f"records={len(records)} resident={len(resident)} invalidated={len(spill_regime)}")
    print(f"conditions={len(summaries)}")
    print(f"absolute maximum: {envelopes['absolute_tested_maximum']}")
    print(f"routine envelope: {envelopes['recommended_routine_envelope']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
