"""Phase-0 orchestration: identity validation and the smoke generation.

Torch is imported lazily so this module (and its records) stay importable in
CPU-only environments; the actual generation only runs on the GPU runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from resonance_asri.reference.config import (
    EXPECTED_SMOKE_RESPONSE,
    SMOKE_MAX_NEW_TOKENS,
    SMOKE_PROMPT,
)
from resonance_asri.reference.validation import (
    Phase0Facts,
    checks_as_dicts,
    evaluate_phase0,
    verdict,
)

if TYPE_CHECKING:
    from resonance_asri.reference.loader import LoadedQwenReference

HISTORICAL_SMOKE_REFERENCE = {
    "source": "ASRI-P0 v1 local smoke (C:/AI/ASRI, 2026-08-15)",
    "response": "READY",
    "prompt_tokens": 18,
    "generated_tokens": 2,
    "latency_seconds": 1.2402,
    "peak_allocated_bytes": 8115154432,
    "peak_reserved_bytes": 8128561152,
    "note": "informational comparison only; identity invariants are the PASS criteria",
}


@dataclass(slots=True)
class SmokeRun:
    """One measured smoke generation."""

    prompt: str
    rendered_prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    max_new_tokens: int
    do_sample: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "rendered_prompt": self.rendered_prompt,
            "response": self.response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }


def run_smoke_generation(
    loaded: LoadedQwenReference,
    *,
    prompt: str = SMOKE_PROMPT,
    max_new_tokens: int = SMOKE_MAX_NEW_TOKENS,
) -> SmokeRun:
    """Run one deterministic thinking-disabled smoke generation."""

    import time

    import torch

    tokenizer, model, device = loaded.tokenizer, loaded.model, loaded.device

    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=loaded.config.enable_thinking,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    input_tokens = int(encoded["input_ids"].shape[-1])
    output_ids = generated[0, input_tokens:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    del generated, encoded

    return SmokeRun(
        prompt=prompt,
        rendered_prompt=rendered,
        response=response,
        input_tokens=input_tokens,
        output_tokens=int(output_ids.numel()),
        latency_seconds=elapsed,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def collect_phase0_facts(loaded: LoadedQwenReference, smoke: SmokeRun) -> Phase0Facts:
    """Convert the loaded substrate and smoke run into torch-free facts."""

    import torch

    model = loaded.model
    parameters = list(model.parameters())
    first_device = str(next(iter(parameters)).device)
    all_on_device = all(str(parameter.device) == loaded.config.device for parameter in parameters)
    observed_dtype = str(next(iter(parameters)).dtype)

    return Phase0Facts(
        model_id=loaded.config.model_id,
        requested_revision=loaded.config.revision,
        resolved_revision=loaded.resolved_revision,
        snapshot_path=str(loaded.snapshot_path),
        trust_remote_code=loaded.config.trust_remote_code,
        tokenizer_class=type(loaded.tokenizer).__name__,
        tokenizer_module=type(loaded.tokenizer).__module__,
        model_class=type(model).__name__,
        model_module=type(model).__module__,
        dtype_name=observed_dtype,
        parameter_device=first_device,
        all_parameters_on_device=all_on_device,
        cuda_available=torch.cuda.is_available(),
        enable_thinking=loaded.config.enable_thinking,
        rendered_prompt=smoke.rendered_prompt,
        response_text=smoke.response,
    )


def evaluate_smoke_response(smoke: SmokeRun) -> list[dict[str, Any]]:
    """Informational smoke comparisons against the historical reference."""

    return [
        {
            "comparison": "response_text",
            "observed": smoke.response,
            "historical": HISTORICAL_SMOKE_REFERENCE["response"],
            "matches": smoke.response == EXPECTED_SMOKE_RESPONSE,
        },
        {
            "comparison": "prompt_tokens",
            "observed": smoke.input_tokens,
            "historical": HISTORICAL_SMOKE_REFERENCE["prompt_tokens"],
            "matches": smoke.input_tokens == HISTORICAL_SMOKE_REFERENCE["prompt_tokens"],
        },
        {
            "comparison": "generated_tokens",
            "observed": smoke.output_tokens,
            "historical": HISTORICAL_SMOKE_REFERENCE["generated_tokens"],
            "matches": smoke.output_tokens == HISTORICAL_SMOKE_REFERENCE["generated_tokens"],
        },
    ]


def write_phase0_artifacts(
    out_dir: str | Path,
    *,
    environment: dict[str, Any],
    loaded: LoadedQwenReference,
    smoke: SmokeRun,
) -> dict[str, Any]:
    """Write environment.json and smoke.json; return the smoke payload."""

    config = loaded.config
    facts = collect_phase0_facts(loaded, smoke)
    checks = evaluate_phase0(facts)
    payload: dict[str, Any] = {
        "phase": "phase0",
        "verdict": verdict(checks),
        "reference": config.as_public_dict(),
        "identity": {
            "resolved_revision": loaded.resolved_revision,
            "snapshot_path": str(loaded.snapshot_path),
            "tokenizer_class": facts.tokenizer_class,
            "tokenizer_module": facts.tokenizer_module,
            "model_class": facts.model_class,
            "model_module": facts.model_module,
        },
        "smoke": smoke.as_dict(),
        "checks": checks_as_dicts(checks),
        "historical_reference": HISTORICAL_SMOKE_REFERENCE,
        "smoke_vs_historical": evaluate_smoke_response(smoke),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(
        json.dumps(environment, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (out / "smoke.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return payload
