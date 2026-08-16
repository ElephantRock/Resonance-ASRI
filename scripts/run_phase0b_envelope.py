#!/usr/bin/env python3
"""Run the ASRI Phase-0B Qwen3-4B inference-envelope sweep.

Loads the frozen substrate once, verifies identity, then measures the
prompt-token x max-new-token grid with one warm-up and three synchronized
measured repetitions per condition. Artifacts are checkpointed after every
condition so a terminated sweep still leaves valid partial evidence. OOM
policy: record the failure, run documented CUDA recovery, and skip the
remaining larger-generation conditions for that prompt size rather than
retrying an unsafe condition. A non-OOM CUDA error terminates the sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from resonance_asri.reference import envelope
from resonance_asri.reference.environment import capture_environment, write_environment_json
from resonance_asri.reference.loader import load_frozen_qwen_reference

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "s0" / "artifacts" / "phase0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repetitions", type=int, default=envelope.MEASURED_REPETITIONS)
    parser.add_argument(
        "--prompt-targets",
        default=",".join(str(value) for value in envelope.PROMPT_TOKEN_TARGETS),
        help="Comma-separated prompt-token targets.",
    )
    parser.add_argument(
        "--max-new-targets",
        default=",".join(str(value) for value in envelope.GENERATION_TOKEN_TARGETS),
        help="Comma-separated max_new_tokens targets.",
    )
    return parser.parse_args()


def verify_identity(loaded: Any) -> None:
    import torch

    parameters = list(loaded.model.parameters())
    assert loaded.resolved_revision == loaded.config.revision, "snapshot revision drifted"
    assert str(next(iter(parameters)).device) == loaded.config.device, "model off device"
    assert str(next(iter(parameters)).dtype) == loaded.dtype_name, "dtype drifted"
    assert torch.cuda.is_available(), "CUDA unavailable at sweep start"


def checkpoint(
    out_dir: Path,
    records: list[dict[str, Any]],
    environment: dict[str, Any],
    reference: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rewrite all artifacts from current records (crash-tolerant)."""

    summaries = envelope.aggregate_runs(records)
    total_vram = int(environment["gpu"]["total_vram_bytes"])
    envelopes = envelope.determine_envelopes(summaries, total_vram_bytes=total_vram)
    payload = {
        "phase": "phase0b",
        "experiment_id": envelope.EXPERIMENT_ID,
        "reference": reference,
        "methodology": metadata,
        "conditions": summaries,
        "envelopes": envelopes,
        "environment_ref": "environment.json",
    }
    envelope.write_runs_jsonl(out_dir / "runs.jsonl", records)
    envelope.write_summary_csv(out_dir / "summary.csv", summaries)
    envelope.write_summary_json(out_dir / "summary.json", payload)
    write_environment_json(out_dir / "environment.json", environment)
    return summaries


def main() -> int:
    args = parse_args()
    prompt_targets = [int(value) for value in args.prompt_targets.split(",")]
    max_new_targets = [int(value) for value in args.max_new_targets.split(",")]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase0b] output={out_dir}", flush=True)
    environment = capture_environment(REPO_ROOT)
    git = environment["git"]

    loaded = load_frozen_qwen_reference()
    verify_identity(loaded)
    tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device

    reference = {
        **loaded.config.as_public_dict(),
        "resolved_revision": loaded.resolved_revision,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    write_environment_json(out_dir / "environment.json", environment)

    metadata: dict[str, Any] = {
        "warmup_runs_per_condition": envelope.WARMUP_RUNS_PER_CONDITION,
        "measured_repetitions_per_condition": args.repetitions,
        "batch_size": 1,
        "prompt_targets": prompt_targets,
        "max_new_targets": max_new_targets,
        "document_seed": envelope.DOCUMENT_SEED,
        "generation_settings": {
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "latency_definition": "full generate call bracketed by torch.cuda.synchronize",
        "ttft": None,
        "skipped_conditions": [],
        "terminated_early": False,
        "termination_reason": None,
    }

    records: list[dict[str, Any]] = []

    # Global warm-up generation so the first measured condition is not the
    # process's very first CUDA work.
    warmup_planned = envelope.plan_prompt(
        tokenizer, target_prompt_tokens=64, seed=envelope.DOCUMENT_SEED
    )
    warmup_record = envelope.base_record(
        repetition=0,
        planned=warmup_planned,
        max_new_tokens=8,
        reference=reference,
        git=git,
        warmup=True,
    )
    print("[phase0b] global warm-up...", flush=True)
    envelope.measure_repetition(
        model=model,
        tokenizer=tokenizer,
        device=device,
        rendered=warmup_planned.rendered,
        max_new_tokens=8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        record=warmup_record,
    )
    print(f"[phase0b] warm-up success={warmup_record['success']}", flush=True)

    try:
        for prompt_target in sorted(prompt_targets):
            planned = envelope.plan_prompt(
                tokenizer,
                target_prompt_tokens=prompt_target,
                seed=envelope.DOCUMENT_SEED + prompt_target,
            )
            print(
                f"[phase0b] prompt target={prompt_target} "
                f"actual={planned.actual_prompt_tokens} tokens",
                flush=True,
            )
            skip_rest = False

            for max_new_target in sorted(max_new_targets):
                condition = f"({prompt_target}, {max_new_target})"
                if skip_rest:
                    metadata["skipped_conditions"].append(
                        {
                            "requested_prompt_tokens": prompt_target,
                            "max_new_tokens": max_new_target,
                            "reason": "skipped_after_oom",
                        }
                    )
                    print(f"[phase0b] skip {condition} (skipped_after_oom)", flush=True)
                    continue

                condition_records: list[dict[str, Any]] = []
                oom_seen = False
                fatal_error: Exception | None = None
                try:
                    for repetition in range(1 + args.repetitions):
                        is_warmup = repetition == 0
                        record = envelope.base_record(
                            repetition=0 if is_warmup else repetition,
                            planned=planned,
                            max_new_tokens=max_new_target,
                            reference=reference,
                            git=git,
                            warmup=is_warmup,
                        )
                        envelope.measure_repetition(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            rendered=planned.rendered,
                            max_new_tokens=max_new_target,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            record=record,
                        )
                        if not is_warmup:
                            condition_records.append(record)
                        if record["success"] is False:
                            if envelope.is_oom_record(record):
                                oom_seen = True
                            else:
                                fatal_error = RuntimeError(
                                    f"{record.get('error_type')}: {record.get('error_message')}"
                                )
                            print(
                                f"[phase0b] {condition} rep={repetition} FAILED "
                                f"type={record.get('error_type')}",
                                flush=True,
                            )
                            break
                        latency = record.get("latency_seconds")
                        print(
                            f"[phase0b] {condition} "
                            f"{'warmup' if is_warmup else f'rep={repetition}'} "
                            f"latency={latency:.3f}s "
                            f"gen={record.get('actual_generated_tokens')} "
                            f"peak_alloc={record.get('peak_allocated_bytes')}",
                            flush=True,
                        )
                finally:
                    records.extend(condition_records)

                if oom_seen:
                    envelope.cuda_recovery()
                    skip_rest = True

                checkpoint(out_dir, records, environment, reference, metadata)

                if fatal_error is not None:
                    # Non-OOM CUDA failures may leave the context unreliable;
                    # hardware safety outranks completing every grid point.
                    raise fatal_error

        metadata["terminated_early"] = False
    except KeyboardInterrupt:
        metadata["terminated_early"] = True
        metadata["termination_reason"] = "keyboard_interrupt"
    except Exception as exc:  # noqa: BLE001 - record, checkpoint, then stop the sweep
        metadata["terminated_early"] = True
        metadata["termination_reason"] = f"{type(exc).__name__}: {exc}"[:500]
        print(f"[phase0b] terminating sweep: {metadata['termination_reason']}", flush=True)

    summaries = checkpoint(out_dir, records, environment, reference, metadata)
    envelopes = envelope.determine_envelopes(
        summaries,
        total_vram_bytes=int(environment["gpu"]["total_vram_bytes"]),
    )

    total_measured = len(records)
    failures = sum(1 for record in records if record["success"] is False)
    print(
        f"[phase0b] done: conditions={len(summaries)} measured_runs={total_measured} "
        f"failures={failures} terminated_early={metadata['terminated_early']}",
        flush=True,
    )
    print(f"[phase0b] absolute maximum: {envelopes['absolute_tested_maximum']}", flush=True)
    print(
        f"[phase0b] routine envelope: {envelopes['recommended_routine_envelope']}",
        flush=True,
    )
    return 0
