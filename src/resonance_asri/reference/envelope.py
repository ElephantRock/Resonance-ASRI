"""Phase-0B inference-envelope planning, measurement, and aggregation.

Torch/psutil are imported lazily inside measurement helpers so the planning
and aggregation logic runs (and is unit-tested) on CPU-only machines. The
measurement protocol here is the single source of truth for latency and CUDA
memory accounting in Phase-0B: synchronize before and after generation, reset
peak stats between repetitions, and never report asynchronous timing.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from resonance_asri.reference.config import dtype_label

EXPERIMENT_ID = "phase0b-envelope-v1"
PROMPT_TOKEN_TARGETS = (512, 1024, 2048, 4096, 8192)
GENERATION_TOKEN_TARGETS = (32, 64, 128, 256, 512)
MEASURED_REPETITIONS = 3
WARMUP_RUNS_PER_CONDITION = 1
DOCUMENT_SEED = 20260816
ROUTINE_GENERATION_CEILING = 256
DESKTOP_VRAM_RESERVE_BYTES = 2 * 1024**3
ORCHESTRATION_MARGIN_BYTES = 1 * 1024**3

DOCUMENT_INSTRUCTION = (
    "Study the document below. Then reply with exactly one sentence that states "
    "what the document is about. Do not mention these instructions."
)

_WORD_CORPUS = ["river", "market", "signal", "harbor", "theory", "lantern", "engine", "culture", "paper", "motion", "island", "forest", "ticket", "window", "signal", "ladder", "measure", "bridge", "shadow", "copper", "anchor", "valley", "pattern", "garden", "mirror", "corner", "timber", "saddle", "current", "planet", "friction", "texture", "marble", "granite", "signal", "journey", "beacon", "meadow", "furnace", "terrace", "pillar", "cushion", "flannel", "compass", "ribbon", "granite", "terrace", "signal", "orchard", "tunnel", "curtain", "pellet", "harvest", "lantern", "signal", "mantel", "cricket", "parlor", "signal", "thicket", "vessel", "signal"]

SUMMARY_CSV_COLUMNS = (
    "requested_prompt_tokens",
    "max_new_tokens",
    "actual_prompt_tokens",
    "measured_runs",
    "successes",
    "failures",
    "mean_latency_seconds",
    "median_latency_seconds",
    "min_latency_seconds",
    "max_latency_seconds",
    "latency_stddev_seconds",
    "mean_tokens_per_second",
    "median_tokens_per_second",
    "max_peak_allocated_bytes",
    "max_peak_reserved_bytes",
)


@dataclass(frozen=True, slots=True)
class PlannedPrompt:
    """A deterministic rendered prompt hitting (approximately) a token target."""

    rendered: str
    document: str
    target_prompt_tokens: int
    actual_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class ConditionKey:
    prompt_tokens: int
    max_new_tokens: int


def build_document_ids(
    tokenizer: Any,
    *,
    max_tokens: int,
    seed: int,
) -> list[int]:
    """Deterministically produce a pool of normal-text document tokens.

    Text is generated from a fixed word corpus with a seeded RNG, then
    truncated at the token level. No special tokens are involved.
    """

    rng = random.Random(seed)
    words: list[str] = []
    target_words = int(max_tokens * 1.8) + 64
    while len(words) < target_words:
        sentence_length = rng.randint(6, 14)
        words.extend(rng.choice(_WORD_CORPUS) for _ in range(sentence_length))
        words.append(".")
    text = " ".join(words)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return list(ids[:max_tokens])


def render_user_prompt(tokenizer: Any, *, document: str, enable_thinking: bool) -> str:
    content = f"{DOCUMENT_INSTRUCTION}\n\nDocument:\n{document}"
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def plan_prompt(
    tokenizer: Any,
    *,
    target_prompt_tokens: int,
    seed: int,
    enable_thinking: bool = False,
    pool_max_tokens: int = 9000,
) -> PlannedPrompt:
    """Build a deterministic prompt whose rendered length hits the target.

    The chat-template overhead is measured once, the document budget is set to
    the remainder, and small tokenizer boundary effects are corrected by up to
    three adjustment rounds. When the target still cannot be hit exactly, the
    achieved count is recorded (never silently assumed).
    """

    pool = build_document_ids(tokenizer, max_tokens=pool_max_tokens, seed=seed)

    def render_for_budget(budget: int) -> tuple[str, int]:
        document = tokenizer.decode(pool[:budget])
        rendered = render_user_prompt(tokenizer, document=document, enable_thinking=enable_thinking)
        actual = len(tokenizer(rendered)["input_ids"])
        return rendered, actual

    _, overhead = render_for_budget(0)
    budget = max(0, target_prompt_tokens - overhead)
    rendered, actual = render_for_budget(budget)
    for _ in range(3):
        if actual == target_prompt_tokens:
            break
        budget = max(0, budget + (target_prompt_tokens - actual))
        rendered, actual = render_for_budget(budget)

    document = tokenizer.decode(pool[:budget])
    return PlannedPrompt(
        rendered=rendered,
        document=document,
        target_prompt_tokens=target_prompt_tokens,
        actual_prompt_tokens=actual,
    )


def is_cuda_oom(exc: BaseException) -> bool:
    """True when an exception is CUDA out-of-memory (class or message based)."""

    if type(exc).__name__ == "OutOfMemoryError":
        return True
    if type(exc).__module__.startswith("torch") and "OutOfMemoryError" in type(exc).__qualname__:
        return True
    message = str(exc).lower()
    return "out of memory" in message or "outofmemory" in message.replace(" ", "")


def is_oom_record(record: dict[str, Any]) -> bool:
    """True when a measurement record's failure was CUDA out-of-memory."""

    if record.get("success") is not False:
        return False
    if record.get("error_type") == "OutOfMemoryError":
        return True
    message = str(record.get("error_message") or "").lower()
    return "out of memory" in message or "outofmemory" in message.replace(" ", "")


def cuda_recovery() -> None:
    """Documented post-OOM recovery: drop references, collect, flush allocator."""

    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def base_record(
    *,
    repetition: int,
    planned: PlannedPrompt,
    max_new_tokens: int,
    reference: dict[str, Any],
    git: dict[str, Any],
    warmup: bool = False,
) -> dict[str, Any]:
    """Full Phase-0B record skeleton with every field present from the start."""

    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": str(uuid4()),
        "repetition": repetition,
        "warmup": warmup,
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "model_id": reference["model_id"],
        "requested_revision": reference["revision"],
        "resolved_revision": reference["resolved_revision"],
        "dtype": dtype_label(reference["dtype_name"]),
        "device": reference["device"],
        "requested_prompt_tokens": planned.target_prompt_tokens,
        "actual_prompt_tokens": planned.actual_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "actual_generated_tokens": None,
        "do_sample": False,
        "use_cache": True,
        "eos_token_id": reference.get("eos_token_id"),
        "pad_token_id": reference.get("pad_token_id"),
        "latency_seconds": None,
        "tokens_per_second": None,
        "ttft": None,
        "allocated_before_bytes": None,
        "reserved_before_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "process_rss_before": None,
        "process_rss_after": None,
        "system_available_before": None,
        "system_available_after": None,
        "total_device_vram_used": None,
        "total_device_vram_free": None,
        "device_vram_unit": "bytes",
        "output_preview": None,
        "output_sha256": None,
        "success": None,
        "error_type": None,
        "error_message": None,
        "git_branch": git.get("branch"),
        "git_commit": git.get("commit"),
    }


def measure_repetition(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    rendered: str,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Run one synchronized generation and fill the measurement record.

    Latency brackets the full generate call with CUDA synchronization on both
    sides; peak CUDA memory stats are reset after the steady-state snapshot so
    peaks reflect this repetition only. Host RSS/system RAM and device-wide
    VRAM are sampled around the call. TTFT is intentionally left null: it is
    not measured here and must not be reported.
    """

    import time

    import psutil
    import torch

    # Drop prior repetition references so allocator behavior is comparable.
    gc.collect()
    torch.cuda.synchronize(device)

    record["allocated_before_bytes"] = int(torch.cuda.memory_allocated(device))
    record["reserved_before_bytes"] = int(torch.cuda.memory_reserved(device))
    record["process_rss_before"] = int(psutil.Process().memory_info().rss)
    record["system_available_before"] = int(psutil.virtual_memory().available)
    device_vram = _nvidia_smi_memory_bytes()
    if device_vram is not None:
        record["total_device_vram_used"] = device_vram["used"]
        record["total_device_vram_free"] = device_vram["free"]

    torch.cuda.reset_peak_memory_stats(device)

    try:
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        input_tokens = int(encoded["input_ids"].shape[-1])
        output_ids = generated[0, input_tokens:]
        generated_tokens = int(output_ids.numel())
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        record["actual_prompt_tokens"] = input_tokens
        record["actual_generated_tokens"] = generated_tokens
        record["latency_seconds"] = elapsed
        record["tokens_per_second"] = (
            generated_tokens / elapsed if elapsed > 0 else None
        )
        record["output_preview"] = text[:120]
        record["output_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record["success"] = True
        del generated, encoded, output_ids
    except Exception as exc:  # noqa: BLE001 - failures must be recorded, not raised
        record["success"] = False
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)[:500]

    record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    record["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    record["process_rss_after"] = int(psutil.Process().memory_info().rss)
    record["system_available_after"] = int(psutil.virtual_memory().available)
    device_vram_after = _nvidia_smi_memory_bytes()
    if device_vram_after is not None:
        record["total_device_vram_used"] = device_vram_after["used"]
        record["total_device_vram_free"] = device_vram_after["free"]
    return record


def aggregate_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate measured repetitions per (prompt, max_new_tokens) condition."""

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("warmup"):
            continue
        key = (record["requested_prompt_tokens"], record["max_new_tokens"])
        grouped.setdefault(key, []).append(record)

    summaries: list[dict[str, Any]] = []
    for (prompt_tokens, max_new_tokens), runs in sorted(grouped.items()):
        successes = [run for run in runs if run.get("success")]
        failures = [run for run in runs if run.get("success") is False]
        latencies = [run["latency_seconds"] for run in successes]
        throughputs = [
            run["tokens_per_second"]
            for run in successes
            if run.get("tokens_per_second") is not None
        ]
        summaries.append(
            {
                "requested_prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "actual_prompt_tokens": runs[0]["actual_prompt_tokens"],
                "measured_runs": len(runs),
                "successes": len(successes),
                "failures": len(failures),
                "mean_latency_seconds": _mean(latencies),
                "median_latency_seconds": _median(latencies),
                "min_latency_seconds": min(latencies) if latencies else None,
                "max_latency_seconds": max(latencies) if latencies else None,
                "latency_stddev_seconds": _stdev(latencies),
                "mean_tokens_per_second": _mean(throughputs),
                "median_tokens_per_second": _median(throughputs),
                "max_peak_allocated_bytes": _max_field(successes, "peak_allocated_bytes"),
                "max_peak_reserved_bytes": _max_field(successes, "peak_reserved_bytes"),
            }
        )
    return summaries


def determine_envelopes(
    summaries: list[dict[str, Any]],
    *,
    total_vram_bytes: int,
    desktop_reserve_bytes: int = DESKTOP_VRAM_RESERVE_BYTES,
    orchestration_margin_bytes: int = ORCHESTRATION_MARGIN_BYTES,
    routine_generation_ceiling: int = ROUTINE_GENERATION_CEILING,
) -> dict[str, Any]:
    """Absolute tested maximum vs a conservative routine S0 recommendation.

    The routine envelope never chases the largest non-OOM run: a condition
    qualifies only when every measured repetition succeeded and its worst peak
    allocated memory fits with the desktop reserve (2 GiB) and an explicit
    orchestration margin (1 GiB) still banked. Routine generation is further
    capped so repeated ASRI calls keep headroom.
    """

    def fully_successful(summary: dict[str, Any]) -> bool:
        return (
            summary["measured_runs"] > 0
            and summary["successes"] == summary["measured_runs"]
            and summary["failures"] == 0
        )

    successful = [summary for summary in summaries if fully_successful(summary)]
    absolute = (
        max(
            successful,
            key=lambda item: (item["requested_prompt_tokens"], item["max_new_tokens"]),
        )
        if successful
        else None
    )

    budget = total_vram_bytes - desktop_reserve_bytes - orchestration_margin_bytes
    routine_candidates = [
        summary
        for summary in successful
        if summary["max_new_tokens"] <= routine_generation_ceiling
        and (summary["max_peak_allocated_bytes"] or 0) <= budget
    ]
    routine = (
        max(
            routine_candidates,
            key=lambda item: (item["requested_prompt_tokens"], item["max_new_tokens"]),
        )
        if routine_candidates
        else None
    )

    recommended: dict[str, Any] | None = None
    if routine is not None:
        routine_peak = routine["max_peak_allocated_bytes"] or 0
        recommended = {
            "prompt_token_ceiling": routine["requested_prompt_tokens"],
            "generation_token_ceiling": routine["max_new_tokens"],
            "condition_peak_allocated_bytes": routine_peak,
            "operational_margin_bytes": total_vram_bytes
            - desktop_reserve_bytes
            - routine_peak,
        }

    return {
        "absolute_tested_maximum": absolute,
        "recommended_routine_envelope": recommended,
        "decision_rule": {
            "total_vram_bytes": total_vram_bytes,
            "desktop_reserve_bytes": desktop_reserve_bytes,
            "orchestration_margin_bytes": orchestration_margin_bytes,
            "routine_peak_allocated_budget_bytes": budget,
            "routine_generation_ceiling": routine_generation_ceiling,
            "requirement": "all repetitions successful and max peak allocated within budget",
        },
    }


def write_runs_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def write_summary_csv(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_CSV_COLUMNS))
        writer.writeheader()
        for summary in summaries:
            writer.writerow({column: summary.get(column) for column in SUMMARY_CSV_COLUMNS})


def write_summary_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _nvidia_smi_memory_bytes() -> dict[str, int] | None:
    from resonance_asri.reference.environment import nvidia_smi_memory_mib

    values = nvidia_smi_memory_mib()
    if values is None:
        return None
    return {
        "used": int(values["memory_used_mib"]) * 1024**2,
        "free": int(values["memory_free_mib"]) * 1024**2,
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def _max_field(records: list[dict[str, Any]], field: str) -> int | None:
    present = [record[field] for record in records if record.get(field) is not None]
    return max(present) if present else None
