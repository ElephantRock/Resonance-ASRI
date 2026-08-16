#!/usr/bin/env python3
"""Run the ASRI Phase-0B2 forced generation-length residency probe.

Answers one question: what happens to memory and throughput when Qwen3-4B
actually generates 64/128/256/512 tokens? Prompts stay in the known-resident
regime (512 and 1024 tokens); output length is forced with
``min_new_tokens == max_new_tokens == target`` so actual generated length
equals the target. ``min_new_tokens`` is a benchmark-only residency control
and is documented as such in every record; no S0 policy uses it.

Safety: after every repetition the run's peak allocated memory is checked
against the resident budget (total VRAM minus the desktop reserve). On the
first WDDM spill-regime observation the sweep stops immediately and artifacts
are checkpointed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from resonance_asri.reference import envelope
from resonance_asri.reference.environment import capture_environment, write_environment_json
from resonance_asri.reference.loader import load_frozen_qwen_reference

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "s0" / "artifacts" / "phase0b2"

PROMPT_TARGETS = (512, 1024)
GENERATION_TARGETS_BY_PROMPT = {
    512: (64, 128, 256, 512),
    1024: (64, 128, 256),
}
MEASURED_REPETITIONS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repetitions", type=int, default=MEASURED_REPETITIONS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase0b2] output={out_dir}", flush=True)
    environment = capture_environment(REPO_ROOT)
    git = environment["git"]
    write_environment_json(out_dir / "environment.json", environment)

    loaded = load_frozen_qwen_reference()
    tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device
    reference = {
        **loaded.config.as_public_dict(),
        "resolved_revision": loaded.resolved_revision,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    total_vram = int(environment["gpu"]["total_vram_bytes"])
    resident_budget = total_vram - envelope.DESKTOP_VRAM_RESERVE_BYTES

    metadata: dict[str, Any] = {
        "purpose": "generation-side KV-cache residency probe",
        "prompt_targets": list(PROMPT_TARGETS),
        "generation_targets": {
            str(key): list(values) for key, values in GENERATION_TARGETS_BY_PROMPT.items()
        },
        "measured_repetitions_per_condition": args.repetitions,
        "warmup_runs_per_condition": envelope.WARMUP_RUNS_PER_CONDITION,
        "batch_size": 1,
        "document_seed": envelope.DOCUMENT_SEED,
        "generation_settings": {
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "min_new_tokens": "equals max_new_tokens (forces target output length)",
            "note": "min_new_tokens is a benchmark-only residency control; no S0 policy uses it",
        },
        "resident_budget_bytes": resident_budget,
        "stop_rule": "first record with peak_allocated_bytes > resident budget stops the sweep",
        "terminated_early": False,
        "termination_reason": None,
        "spill_first_observed": None,
    }

    records: list[dict[str, Any]] = []

    def checkpoint() -> None:
        resident, spill_regime = envelope.classify_residency(
            records, total_vram_bytes=total_vram
        )
        summaries = envelope.aggregate_runs(resident)
        payload = {
            "phase": "phase0b2",
            "experiment_id": envelope.PHASE0B2_EXPERIMENT_ID,
            "reference": reference,
            "methodology": metadata,
            "conditions": summaries,
            "invalidated_runs": [
                {
                    "run_id": record["run_id"],
                    "requested_prompt_tokens": record["requested_prompt_tokens"],
                    "max_new_tokens": record["max_new_tokens"],
                    "repetition": record["repetition"],
                    "peak_allocated_bytes": record["peak_allocated_bytes"],
                    "reason": "wddm_shared_memory_spill_regime",
                }
                for record in spill_regime
            ],
            "environment_ref": "environment.json",
        }
        envelope.write_runs_jsonl(
            out_dir / "runs.jsonl",
            [record for record in records if not record.get("warmup")],
        )
        envelope.write_summary_csv(out_dir / "summary.csv", summaries)
        envelope.write_summary_json(out_dir / "summary.json", payload)

    # Global warm-up generation (short, resident).
    warm_planned = envelope.plan_prompt(
        tokenizer, target_prompt_tokens=64, seed=envelope.DOCUMENT_SEED
    )
    warm_record = envelope.base_record(
        repetition=0,
        planned=warm_planned,
        max_new_tokens=8,
        reference=reference,
        git=git,
        warmup=True,
        experiment_id=envelope.PHASE0B2_EXPERIMENT_ID,
    )
    print("[phase0b2] global warm-up...", flush=True)
    envelope.measure_repetition(
        model=model,
        tokenizer=tokenizer,
        device=device,
        rendered=warm_planned.rendered,
        max_new_tokens=8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        record=warm_record,
    )
    print(f"[phase0b2] warm-up success={warm_record['success']}", flush=True)

    spill_detected = False
    for prompt_target in PROMPT_TARGETS:
        if spill_detected:
            break
        planned = envelope.plan_prompt(
            tokenizer,
            target_prompt_tokens=prompt_target,
            seed=envelope.DOCUMENT_SEED + prompt_target,
        )
        print(
            f"[phase0b2] prompt target={prompt_target} "
            f"actual={planned.actual_prompt_tokens} tokens",
            flush=True,
        )
        for generation_target in GENERATION_TARGETS_BY_PROMPT[prompt_target]:
            if spill_detected:
                break
            condition = f"({prompt_target}, gen~{generation_target})"
            condition_records: list[dict[str, Any]] = []
            try:
                for repetition in range(1 + args.repetitions):
                    is_warmup = repetition == 0
                    record = envelope.base_record(
                        repetition=0 if is_warmup else repetition,
                        planned=planned,
                        max_new_tokens=generation_target,
                        reference=reference,
                        git=git,
                        warmup=is_warmup,
                        experiment_id=envelope.PHASE0B2_EXPERIMENT_ID,
                        min_new_tokens=generation_target,
                    )
                    envelope.measure_repetition(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        rendered=planned.rendered,
                        max_new_tokens=generation_target,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        record=record,
                        min_new_tokens=generation_target,
                    )
                    if not is_warmup:
                        condition_records.append(record)

                    peak = record.get("peak_allocated_bytes")
                    if record["success"] is False:
                        print(
                            f"[phase0b2] {condition} rep={repetition} FAILED "
                            f"type={record.get('error_type')}",
                            flush=True,
                        )
                        spill_detected = True  # treat any failure as stop condition
                        break
                    if peak is not None and peak > resident_budget:
                        spill_detected = True
                        metadata["spill_first_observed"] = {
                            "condition": [prompt_target, generation_target],
                            "repetition": repetition,
                            "warmup": is_warmup,
                            "peak_allocated_bytes": peak,
                        }
                        print(
                            f"[phase0b2] SPILL DETECTED {condition}: peak={peak} "
                            f"> budget={resident_budget}; stopping immediately",
                            flush=True,
                        )
                        break
                    latency = record.get("latency_seconds")
                    print(
                        f"[phase0b2] {condition} "
                        f"{'warmup' if is_warmup else f'rep={repetition}'} "
                        f"latency={latency:.3f}s "
                        f"gen={record.get('actual_generated_tokens')} "
                        f"peak_alloc={peak}",
                        flush=True,
                    )
            finally:
                records.extend(condition_records)
            checkpoint()
            if spill_detected:
                break

    if spill_detected:
        envelope.cuda_recovery()
        metadata["terminated_early"] = True
        metadata["termination_reason"] = "wddm_spill_regime_detected"

    checkpoint()

    resident, _ = envelope.classify_residency(records, total_vram_bytes=total_vram)
    summaries = envelope.aggregate_runs(resident)
    print(
        f"[phase0b2] done: measured_runs={len(records)} "
        f"valid_conditions={len(summaries)} spill_stop={spill_detected}",
        flush=True,
    )
    for summary in summaries:
        print(
            f"[phase0b2] condition ({summary['requested_prompt_tokens']}, "
            f"gen~{summary['max_new_tokens']}): mean_latency="
            f"{summary['mean_latency_seconds']:.3f}s "
            f"mean_tps={summary['mean_tokens_per_second']:.2f} "
            f"max_peak_alloc={summary['max_peak_allocated_bytes']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
