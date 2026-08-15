from __future__ import annotations

from dataclasses import dataclass, field

from resonance_asri.contracts import ExecutionRequest
from resonance_asri.controller import FixedDeepPolicy, FixedShallowPolicy
from resonance_asri.providers import ProviderResponse
from resonance_asri.runtime import ASRIRuntime


@dataclass
class FakeProvider:
    model_id: str = "fake-model"
    purposes: list[str] = field(default_factory=list)

    def complete(self, *, prompt: str, purpose: str) -> ProviderResponse:
        assert prompt
        self.purposes.append(purpose)
        return ProviderResponse(
            text=f"answer-{len(self.purposes)}",
            model_id=self.model_id,
            input_tokens=10,
            output_tokens=2,
            estimated_cost_usd=0.001,
        )


def test_shallow_runtime_emits_single_call_receipt() -> None:
    provider = FakeProvider()
    runtime = ASRIRuntime(provider=provider, policy=FixedShallowPolicy())

    receipt = runtime.run(ExecutionRequest(request_id="r1", prompt="simple task"))

    assert receipt.answer == "answer-1"
    assert receipt.usage.provider_calls == 1
    assert receipt.usage.reasoning_iterations == 1
    assert receipt.usage.input_tokens == 10
    assert receipt.usage.output_tokens == 2
    assert receipt.usage.estimated_cost_usd == 0.001
    assert provider.purposes == ["answer"]


def test_deep_runtime_accounts_for_refinement_specialist_and_verifier() -> None:
    provider = FakeProvider()
    runtime = ASRIRuntime(provider=provider, policy=FixedDeepPolicy())

    receipt = runtime.run(ExecutionRequest(request_id="r2", prompt="difficult task"))

    assert receipt.answer == "answer-7"
    assert receipt.usage.provider_calls == 7
    assert receipt.usage.reasoning_iterations == 4
    assert receipt.usage.verifier_count == 1
    assert receipt.specialists_activated == ("general-reviewer",)
    assert receipt.metadata["nominal_compute_units"] == "7"
    assert provider.purposes == [
        "answer",
        "refine",
        "refine",
        "refine",
        "specialist:general-reviewer",
        "specialist-synthesis",
        "verify",
    ]
