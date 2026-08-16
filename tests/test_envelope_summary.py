import csv
import json
from pathlib import Path

from resonance_asri.reference import envelope

GIB = 1024**3

REQUIRED_RUN_FIELDS = {
    "experiment_id",
    "run_id",
    "repetition",
    "timestamp",
    "model_id",
    "requested_revision",
    "resolved_revision",
    "dtype",
    "device",
    "requested_prompt_tokens",
    "actual_prompt_tokens",
    "max_new_tokens",
    "actual_generated_tokens",
    "do_sample",
    "latency_seconds",
    "tokens_per_second",
    "allocated_before_bytes",
    "reserved_before_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "process_rss_before",
    "process_rss_after",
    "system_available_before",
    "system_available_after",
    "total_device_vram_used",
    "total_device_vram_free",
    "success",
    "error_type",
    "error_message",
    "git_branch",
    "git_commit",
}


class FakeWordTokenizer:
    """Whitespace tokenizer: one token per word, stable int ids, chat overhead."""

    eos_token_id = 151645
    pad_token_id = 151643

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors=None):
        ids = [self._word_id(word) for word in text.split()]
        return {"input_ids": ids}

    def decode(self, ids) -> str:
        reverse = {value: key for key, value in self._vocab.items()}
        return " ".join(reverse[int(token_id)] for token_id in ids)

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        return "CHAT_START user " + messages[0]["content"] + " CHAT_END assistant_start"

    def _word_id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = 1000 + len(self._vocab)
        return self._vocab[word]


def make_record(
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    repetition: int,
    latency: float | None = 1.0,
    tps: float | None = 10.0,
    peak_alloc: int | None = 8 * GIB,
    success: bool = True,
    error_type: str | None = None,
    error_message: str | None = None,
    warmup: bool = False,
) -> dict:
    return {
        "experiment_id": envelope.EXPERIMENT_ID,
        "requested_prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "repetition": repetition,
        "warmup": warmup,
        "latency_seconds": latency,
        "tokens_per_second": tps,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": (peak_alloc + 128 * 1024**2) if peak_alloc else None,
        "success": success,
        "error_type": error_type,
        "error_message": error_message,
    }


def test_plan_prompt_hits_exact_token_target_with_fake_tokenizer() -> None:
    tokenizer = FakeWordTokenizer()
    planned = envelope.plan_prompt(tokenizer, target_prompt_tokens=64, seed=1)

    empty_content = f"{envelope.DOCUMENT_INSTRUCTION}\n\nDocument:\n"
    overhead_render = tokenizer.apply_chat_template(
        [{"role": "user", "content": empty_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    overhead = len(overhead_render.split())

    assert planned.actual_prompt_tokens == 64
    assert planned.target_prompt_tokens == 64
    assert "CHAT_START" in planned.rendered
    assert len(planned.document.split()) == 64 - overhead


def test_document_ids_are_deterministic_and_seed_sensitive() -> None:
    tokenizer = FakeWordTokenizer()
    left = envelope.build_document_ids(tokenizer, max_tokens=50, seed=7)
    right = envelope.build_document_ids(tokenizer, max_tokens=50, seed=7)
    other = envelope.build_document_ids(tokenizer, max_tokens=50, seed=8)
    assert left == right
    assert left != other
    assert len(left) == 50


def test_aggregate_runs_computes_latency_statistics_per_condition() -> None:
    records = [
        make_record(prompt_tokens=512, max_new_tokens=32, repetition=1, latency=1.0, tps=10.0),
        make_record(prompt_tokens=512, max_new_tokens=32, repetition=2, latency=2.0, tps=20.0),
        make_record(prompt_tokens=512, max_new_tokens=32, repetition=3, latency=3.0, tps=30.0),
        make_record(prompt_tokens=512, max_new_tokens=32, repetition=0, warmup=True),
    ]
    summaries = envelope.aggregate_runs(records)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["measured_runs"] == 3  # warm-up excluded
    assert summary["successes"] == 3
    assert summary["mean_latency_seconds"] == 2.0
    assert summary["median_latency_seconds"] == 2.0
    assert summary["min_latency_seconds"] == 1.0
    assert summary["max_latency_seconds"] == 3.0
    assert summary["latency_stddev_seconds"] == 1.0
    assert summary["mean_tokens_per_second"] == 20.0
    assert summary["max_peak_allocated_bytes"] == 8 * GIB


def test_aggregate_runs_counts_failures_and_all_oom_conditions() -> None:
    records = [
        make_record(prompt_tokens=1024, max_new_tokens=64, repetition=1),
        make_record(
            prompt_tokens=1024,
            max_new_tokens=64,
            repetition=2,
            success=False,
            error_type="OutOfMemoryError",
            error_message="CUDA out of memory",
        ),
        make_record(
            prompt_tokens=2048,
            max_new_tokens=512,
            repetition=1,
            success=False,
            error_type="OutOfMemoryError",
            error_message="CUDA out of memory",
        ),
        make_record(
            prompt_tokens=2048,
            max_new_tokens=512,
            repetition=2,
            success=False,
            error_type="OutOfMemoryError",
            error_message="CUDA out of memory",
        ),
    ]
    summaries = envelope.aggregate_runs(records)

    by_condition = {
        (item["requested_prompt_tokens"], item["max_new_tokens"]): item
        for item in summaries
    }
    mixed = by_condition[(1024, 64)]
    assert (mixed["successes"], mixed["failures"]) == (1, 1)
    assert mixed["mean_latency_seconds"] == 1.0

    all_oom = by_condition[(2048, 512)]
    assert (all_oom["successes"], all_oom["failures"]) == (0, 2)
    assert all_oom["mean_latency_seconds"] is None
    assert all_oom["max_peak_allocated_bytes"] is None


def test_classify_residency_separates_spill_regime_runs() -> None:
    total_vram = 12 * GIB
    resident_run = make_record(prompt_tokens=2048, max_new_tokens=512, repetition=1,
                               peak_alloc=9.8 * GIB)
    spill_run = make_record(prompt_tokens=4096, max_new_tokens=32, repetition=1,
                            peak_alloc=13.0 * GIB)
    no_peak_run = make_record(prompt_tokens=512, max_new_tokens=32, repetition=1,
                              peak_alloc=None)

    resident, spill_regime = envelope.classify_residency(
        [resident_run, spill_run, no_peak_run], total_vram_bytes=total_vram
    )

    assert resident_run in resident
    assert no_peak_run in resident
    assert spill_run in spill_regime
    # budget is total VRAM minus the 2 GiB desktop reserve
    assert envelope.classify_residency(
        [make_record(prompt_tokens=1, max_new_tokens=1, repetition=1, peak_alloc=10.1 * GIB)],
        total_vram_bytes=total_vram,
    )[1]


def test_oom_detection_helpers() -> None:
    record = make_record(
        prompt_tokens=512,
        max_new_tokens=32,
        repetition=1,
        success=False,
        error_type="OutOfMemoryError",
        error_message="CUDA out of memory. Tried to allocate ...",
    )
    assert envelope.is_oom_record(record)
    assert envelope.is_cuda_oom(RuntimeError("CUDA out of memory"))

    other = make_record(
        prompt_tokens=512,
        max_new_tokens=32,
        repetition=1,
        success=False,
        error_type="RuntimeError",
        error_message="device-side assert triggered",
    )
    assert not envelope.is_oom_record(other)
    assert not envelope.is_cuda_oom(RuntimeError("device-side assert triggered"))
    assert not envelope.is_oom_record(make_record(prompt_tokens=1, max_new_tokens=1, repetition=1))


def test_determine_envelopes_separates_absolute_maximum_from_routine() -> None:
    total_vram = 12 * GIB
    summaries = [
        # prompt, gen, all reps ok, peak within 9 GiB budget
        make_summary(512, 32, peak=8.0),
        make_summary(2048, 256, peak=8.5),
        # within budget but exceeds routine generation ceiling
        make_summary(2048, 512, peak=8.6),
        # over the memory budget though successful
        make_summary(8192, 512, peak=10.8),
    ]
    envelopes = envelope.determine_envelopes(summaries, total_vram_bytes=total_vram)

    absolute = envelopes["absolute_tested_maximum"]
    assert (absolute["requested_prompt_tokens"], absolute["max_new_tokens"]) == (8192, 512)

    routine = envelopes["recommended_routine_envelope"]
    assert routine["prompt_token_ceiling"] == 2048
    assert routine["generation_token_ceiling"] == 256
    assert routine["operational_margin_bytes"] == total_vram - 2 * GIB - 8.5 * GIB


def make_summary(prompt_tokens: int, max_new: int, *, peak: float) -> dict:
    return {
        "requested_prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new,
        "actual_prompt_tokens": prompt_tokens,
        "measured_runs": 3,
        "successes": 3,
        "failures": 0,
        "mean_latency_seconds": 1.0,
        "max_peak_allocated_bytes": int(peak * GIB),
        "max_peak_reserved_bytes": int(peak * GIB) + 1,
    }


def test_determine_envelopes_with_no_successful_conditions() -> None:
    envelopes = envelope.determine_envelopes([], total_vram_bytes=12 * GIB)
    assert envelopes["absolute_tested_maximum"] is None
    assert envelopes["recommended_routine_envelope"] is None


def test_base_record_contains_every_required_field_and_serializes() -> None:
    planned = envelope.PlannedPrompt(
        rendered="r", document="d", target_prompt_tokens=512, actual_prompt_tokens=511
    )
    record = envelope.base_record(
        repetition=2,
        planned=planned,
        max_new_tokens=64,
        reference={
            "model_id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "resolved_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "dtype_name": "bfloat16",
            "device": "cuda:0",
            "eos_token_id": 151645,
            "pad_token_id": 151643,
        },
        git={"branch": "research/asri-s0", "commit": "4384148"},
    )

    missing = REQUIRED_RUN_FIELDS - set(record)
    assert not missing, f"missing fields: {missing}"
    assert record["do_sample"] is False
    assert record["ttft"] is None
    encoded = json.dumps(record, sort_keys=True)
    assert "phase0b-envelope-v1" in encoded


def test_runs_jsonl_and_summary_csv_round_trip(tmp_path: Path) -> None:
    records = [
        make_record(prompt_tokens=512, max_new_tokens=32, repetition=1),
        make_record(prompt_tokens=512, max_new_tokens=32, repetition=2),
    ]
    summaries = envelope.aggregate_runs(records)

    envelope.write_runs_jsonl(tmp_path / "runs.jsonl", records)
    envelope.write_summary_csv(tmp_path / "summary.csv", summaries)

    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["requested_prompt_tokens"] == 512

    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["requested_prompt_tokens"] == "512"
    assert rows[0]["mean_latency_seconds"] == "1.0"
