#!/usr/bin/env python3
"""Run ASRI Phase-0 reference validation on the frozen Qwen3-4B substrate.

Loads the pinned snapshot, executes the identity invariants plus one smoke
generation, and writes compact JSON artifacts. Exit code 0 only on PASS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from resonance_asri.reference.environment import capture_environment
from resonance_asri.reference.loader import load_frozen_qwen_reference
from resonance_asri.reference.phase0 import (
    run_smoke_generation,
    write_phase0_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research" / "s0" / "artifacts" / "phase0",
        help="Directory for compact validated artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    environment = capture_environment(REPO_ROOT)
    loaded = load_frozen_qwen_reference()
    smoke = run_smoke_generation(loaded)
    payload = write_phase0_artifacts(
        args.output_dir, environment=environment, loaded=loaded, smoke=smoke
    )

    for check in payload["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['check_id']}: {check['detail']}")
    print(f"Phase-0 verdict: {payload['verdict']}")
    print(
        f"smoke response={smoke.response!r} input_tokens={smoke.input_tokens} "
        f"output_tokens={smoke.output_tokens} latency_s={smoke.latency_seconds:.6f}"
    )
    print(
        f"peak_allocated_bytes={smoke.peak_allocated_bytes} "
        f"peak_reserved_bytes={smoke.peak_reserved_bytes}"
    )
    print(f"artifacts={args.output_dir}")
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
