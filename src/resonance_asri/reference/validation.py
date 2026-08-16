"""Phase-0 identity checks and smoke evaluation (model-free).

``Phase0Facts`` is a plain record of everything observed during a Phase-0 run.
``evaluate_phase0`` turns those facts into explicit PASS/FAIL checks. Keeping
both free of torch lets the check logic run (and be tested) on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from resonance_asri.reference.config import (
    FROZEN_QWEN_REFERENCE,
    QwenReferenceConfig,
    dtype_label,
    is_commit_revision,
)

NON_THINKING_MARKER = "<think>\n\n</think>"
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One PASS/FAIL invariant with a human-readable detail line."""

    check_id: str
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class Phase0Facts:
    """Everything Phase-0 observed about the loaded substrate and smoke run."""

    model_id: str
    requested_revision: str
    resolved_revision: str
    snapshot_path: str
    trust_remote_code: bool
    tokenizer_class: str
    tokenizer_module: str
    model_class: str
    model_module: str
    dtype_name: str
    parameter_device: str
    all_parameters_on_device: bool
    cuda_available: bool
    enable_thinking: bool
    rendered_prompt: str
    response_text: str
    notes: tuple[str, ...] = field(default=())


def is_non_thinking_render(rendered: str) -> bool:
    """True when a rendered Qwen3 chat prompt carries the closed empty think block.

    Qwen3's non-thinking mode renders ``<think>\\n\\n</think>\\n\\n`` immediately
    after the assistant header, which instructs the model to skip its thinking
    phase. Thinking mode instead leaves the assistant turn open-ended.
    """

    if NON_THINKING_MARKER not in rendered:
        return False
    return rendered.rstrip().endswith(THINK_CLOSE_TAG)


def contains_think_tags(text: str) -> bool:
    """True when generated text leaks think block tags."""

    return THINK_OPEN_TAG in text or THINK_CLOSE_TAG in text


def evaluate_phase0(
    facts: Phase0Facts,
    *,
    config: QwenReferenceConfig = FROZEN_QWEN_REFERENCE,
) -> list[CheckResult]:
    """Evaluate the twelve Phase-0 identity invariants against observed facts."""

    checks: list[CheckResult] = [
        CheckResult(
            check_id="model_id_matches_frozen",
            name="model_id is Qwen/Qwen3-4B",
            passed=facts.model_id == config.model_id,
            detail=f"observed={facts.model_id!r} expected={config.model_id!r}",
        ),
        CheckResult(
            check_id="requested_revision_is_commit",
            name="requested revision is an exact commit SHA",
            passed=(
                facts.requested_revision == config.revision
                and is_commit_revision(facts.requested_revision)
            ),
            detail=f"requested_revision={facts.requested_revision!r}",
        ),
        CheckResult(
            check_id="resolved_revision_matches_requested",
            name="resolved snapshot revision is exact",
            passed=facts.resolved_revision == facts.requested_revision == config.revision,
            detail=(
                f"resolved={facts.resolved_revision!r} requested={facts.requested_revision!r}"
            ),
        ),
        CheckResult(
            check_id="tokenizer_native_no_remote_code",
            name="tokenizer is a native transformers class (trust_remote_code=False)",
            passed=(
                facts.trust_remote_code is False
                and facts.tokenizer_module.startswith("transformers.")
            ),
            detail=(
                f"class={facts.tokenizer_class} module={facts.tokenizer_module} "
                f"trust_remote_code={facts.trust_remote_code}"
            ),
        ),
        CheckResult(
            check_id="model_native_no_remote_code",
            name="model is a native transformers class (trust_remote_code=False)",
            passed=(
                facts.trust_remote_code is False
                and facts.model_module.startswith("transformers.")
            ),
            detail=(
                f"class={facts.model_class} module={facts.model_module} "
                f"trust_remote_code={facts.trust_remote_code}"
            ),
        ),
        CheckResult(
            check_id="enable_thinking_disabled",
            name="enable_thinking=False",
            passed=facts.enable_thinking is False,
            detail=f"enable_thinking={facts.enable_thinking}",
        ),
        CheckResult(
            check_id="chat_template_non_thinking",
            name="rendered chat template is Qwen non-thinking mode",
            passed=is_non_thinking_render(facts.rendered_prompt),
            detail=_render_detail(facts.rendered_prompt),
        ),
        CheckResult(
            check_id="response_has_no_think_tags",
            name="generated response contains no think tags",
            passed=not contains_think_tags(facts.response_text),
            detail=f"response={facts.response_text[:80]!r}",
        ),
        CheckResult(
            check_id="response_non_empty",
            name="generation is non-empty",
            passed=len(facts.response_text.strip()) > 0,
            detail=f"response={facts.response_text[:80]!r}",
        ),
        CheckResult(
            check_id="cuda_available",
            name="CUDA is available",
            passed=facts.cuda_available is True,
            detail=f"cuda_available={facts.cuda_available}",
        ),
        CheckResult(
            check_id="model_on_reference_device",
            name="model resides on cuda:0",
            passed=(
                facts.parameter_device == config.device
                and facts.all_parameters_on_device is True
            ),
            detail=(
                f"parameter_device={facts.parameter_device!r} "
                f"all_parameters_on_device={facts.all_parameters_on_device}"
            ),
        ),
        CheckResult(
            check_id="dtype_is_bfloat16",
            name="model dtype is BF16",
            passed=facts.dtype_name == dtype_label(config.dtype_name),
            detail=f"dtype={facts.dtype_name!r} expected={dtype_label(config.dtype_name)!r}",
        ),
    ]
    return checks


def verdict(checks: list[CheckResult]) -> str:
    """Overall Phase-0 verdict: PASS only when every invariant passed."""

    return "PASS" if all(check.passed for check in checks) else "FAIL"


def checks_as_dicts(checks: list[CheckResult]) -> list[dict[str, object]]:
    return [check.as_dict() for check in checks]


def _render_detail(rendered: str) -> str:
    marker = (
        "present"
        if NON_THINKING_MARKER in rendered
        else ("open-think" if THINK_OPEN_TAG in rendered else "absent")
    )
    return f"non_thinking_marker={marker} tail={rendered[-40:]!r}"
