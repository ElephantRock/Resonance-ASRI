"""Local Qwen3-4B completion provider for the S0 runtime.

A thin adapter over the frozen reference substrate: non-thinking chat
rendering, deterministic greedy generation, and usage accounting. It contains
no S0 orchestration logic and does not leak tokenizer/model objects to
callers. Torch is imported lazily so the module is importable (and its
normalization logic testable) without a GPU environment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from resonance_asri.providers.base import ProviderResponse


@dataclass(frozen=True, slots=True)
class GenerationStats:
    """Per-call accounting surfaced next to the provider response."""

    latency_ms: float
    input_tokens: int
    output_tokens: int


def build_provider_response(
    *,
    text: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> ProviderResponse:
    """Normalize one raw local generation into the provider contract."""

    return ProviderResponse(
        text=text.strip(),
        model_id=model_id,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        estimated_cost_usd=None,
    )


class QwenLocalProvider:
    """CompletionProvider backed by the frozen local Qwen3-4B reference.

    ``model_id`` should carry the pinned identity, e.g.
    ``Qwen/Qwen3-4B@1cfa9a720891``. Generation is deterministic
    (``do_sample=False``); ``max_new_tokens`` is fixed at construction to keep
    the provider contract free of per-call knobs. Latency is exposed via
    ``last_stats`` because ProviderResponse has no latency field.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        model_id: str,
        device: Any,
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._enable_thinking = enable_thinking
        self.last_stats: GenerationStats | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @classmethod
    def from_reference(
        cls,
        *,
        loaded: Any,
        max_new_tokens: int = 256,
    ) -> QwenLocalProvider:
        """Build a provider from a loaded frozen reference (see loader.py)."""

        return cls(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            model_id=f"{loaded.model_id}@{loaded.resolved_revision[:12]}",
            device=loaded.device,
            max_new_tokens=max_new_tokens,
            enable_thinking=loaded.config.enable_thinking,
        )

    def complete(self, *, prompt: str, purpose: str) -> ProviderResponse:
        del purpose  # routing metadata only; generation settings are frozen
        rendered = self._render(prompt)
        text, input_tokens, output_tokens, latency_ms = self._run_generation(rendered)
        self.last_stats = GenerationStats(
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return build_provider_response(
            text=text,
            model_id=self._model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _render(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )

    def _run_generation(self, rendered: str) -> tuple[str, int, int, float]:
        import torch

        tokenizer = self._tokenizer
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize(self._device)
        latency_ms = (time.perf_counter() - started) * 1000.0

        input_tokens = int(encoded["input_ids"].shape[-1])
        output_ids = generated[0, input_tokens:]
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return text, input_tokens, int(output_ids.numel()), latency_ms
