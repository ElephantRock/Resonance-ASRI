"""Optional frozen-model GPU integration tests.

These load the real Qwen3-4B snapshot and require the local Hugging Face cache
and a CUDA device. They are skipped unless ASRI_RUN_GPU_TESTS is set, so
GitHub Actions (CPU-only, no model download) never triggers them.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ASRI_RUN_GPU_TESTS"),
    reason="set ASRI_RUN_GPU_TESTS=1 to run frozen-model GPU integration tests",
)


def test_phase0_smoke_passes_on_gpu() -> None:
    from resonance_asri.reference.loader import load_frozen_qwen_reference
    from resonance_asri.reference.phase0 import (
        collect_phase0_facts,
        evaluate_smoke_response,
        run_smoke_generation,
    )
    from resonance_asri.reference.validation import evaluate_phase0, verdict

    loaded = load_frozen_qwen_reference()
    smoke = run_smoke_generation(loaded)
    facts = collect_phase0_facts(loaded, smoke)
    checks = evaluate_phase0(facts)

    failed = [check.as_dict() for check in checks if not check.passed]
    assert verdict(checks) == "PASS", failed
    assert smoke.response == "READY"
    assert smoke.output_tokens > 0

    comparisons = evaluate_smoke_response(smoke)
    assert all(item["matches"] for item in comparisons), comparisons


def test_phase0b_envelope_runs_on_gpu(tmp_path) -> None:
    from resonance_asri.reference import envelope
    from resonance_asri.reference.loader import load_frozen_qwen_reference

    loaded = load_frozen_qwen_reference()
    planned = envelope.plan_prompt(
        loaded.tokenizer, target_prompt_tokens=128, seed=envelope.DOCUMENT_SEED
    )
    reference = {
        **loaded.config.as_public_dict(),
        "resolved_revision": loaded.resolved_revision,
        "eos_token_id": loaded.tokenizer.eos_token_id,
        "pad_token_id": loaded.tokenizer.pad_token_id,
    }
    record = envelope.base_record(
        repetition=1,
        planned=planned,
        max_new_tokens=8,
        reference=reference,
        git={"branch": None, "commit": None},
    )
    envelope.measure_repetition(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        device=loaded.device,
        rendered=planned.rendered,
        max_new_tokens=8,
        eos_token_id=loaded.tokenizer.eos_token_id,
        pad_token_id=loaded.tokenizer.pad_token_id,
        record=record,
    )

    assert record["success"] is True
    assert record["actual_prompt_tokens"] == 128
    assert record["actual_generated_tokens"] > 0
    assert record["latency_seconds"] > 0
    assert record["peak_allocated_bytes"] > 0
    assert record["ttft"] is None
