import pytest

from resonance_asri.providers.base import ProviderResponse
from resonance_asri.providers.qwen_local import (
    GenerationStats,
    QwenLocalProvider,
    build_provider_response,
)


class FakeChatTokenizer:
    """Minimal chat-rendering double for provider tests (no torch)."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        user_text = messages[0]["content"]
        return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


class StubbedProvider(QwenLocalProvider):
    """Provider with the GPU generation step replaced by a stub."""

    def _run_generation(self, rendered: str):
        return "  READY with padding  ", 18, 2, 12.5


def make_provider() -> StubbedProvider:
    return StubbedProvider(
        model=object(),
        tokenizer=FakeChatTokenizer(),
        model_id="Qwen/Qwen3-4B@1cfa9a720891",
        device=None,
        max_new_tokens=256,
    )


def test_provider_complete_returns_normalized_response() -> None:
    provider = make_provider()
    response = provider.complete(prompt="Return exactly the word READY.", purpose="answer")

    assert isinstance(response, ProviderResponse)
    assert response.text == "READY with padding"
    assert response.model_id == "Qwen/Qwen3-4B@1cfa9a720891"
    assert response.input_tokens == 18
    assert response.output_tokens == 2
    assert response.estimated_cost_usd is None


def test_provider_records_latency_and_usage_stats_per_call() -> None:
    provider = make_provider()
    assert provider.last_stats is None

    provider.complete(prompt="task", purpose="refine")

    assert provider.last_stats == GenerationStats(
        latency_ms=12.5, input_tokens=18, output_tokens=2
    )


def test_provider_exposes_model_id_and_frozen_max_new_tokens() -> None:
    provider = make_provider()
    assert provider.model_id == "Qwen/Qwen3-4B@1cfa9a720891"
    assert provider.max_new_tokens == 256


def test_provider_rejects_invalid_max_new_tokens() -> None:
    with pytest.raises(ValueError):
        make_provider_with_tokens(0)


def make_provider_with_tokens(tokens: int) -> QwenLocalProvider:
    return QwenLocalProvider(
        model=object(),
        tokenizer=FakeChatTokenizer(),
        model_id="m",
        device=None,
        max_new_tokens=tokens,
    )


def test_build_provider_response_rejects_negative_usage() -> None:
    with pytest.raises(ValueError):
        build_provider_response(
            text="ok", model_id="m", input_tokens=-1, output_tokens=1
        )
    with pytest.raises(ValueError):
        build_provider_response(
            text="ok", model_id="m", input_tokens=1, output_tokens=-1
        )
