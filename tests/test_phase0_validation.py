import json
from dataclasses import replace

from resonance_asri.reference.validation import (
    Phase0Facts,
    contains_think_tags,
    evaluate_phase0,
    is_non_thinking_render,
    verdict,
)

NON_THINKING_RENDER = (
    "<|im_start|>user\nReturn exactly the word READY.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def valid_facts() -> Phase0Facts:
    return Phase0Facts(
        model_id="Qwen/Qwen3-4B",
        requested_revision="1cfa9a7208912126459214e8b04321603b3df60c",
        resolved_revision="1cfa9a7208912126459214e8b04321603b3df60c",
        snapshot_path="C:/huggingface_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7",
        trust_remote_code=False,
        tokenizer_class="Qwen2Tokenizer",
        tokenizer_module="transformers.models.qwen2.tokenization_qwen2",
        model_class="Qwen3ForCausalLM",
        model_module="transformers.models.qwen3.modeling_qwen3",
        dtype_name="torch.bfloat16",
        parameter_device="cuda:0",
        all_parameters_on_device=True,
        cuda_available=True,
        enable_thinking=False,
        rendered_prompt=NON_THINKING_RENDER,
        response_text="READY",
    )


def check_ids(results) -> set[str]:
    return {result.check_id for result in results}


def failing_ids(results) -> set[str]:
    return {result.check_id for result in results if not result.passed}


def test_all_invariants_pass_for_valid_facts() -> None:
    results = evaluate_phase0(valid_facts())
    assert len(results) == 12
    assert not failing_ids(results)
    assert verdict(results) == "PASS"


def test_wrong_model_id_fails_identity_check() -> None:
    results = evaluate_phase0(replace(valid_facts(), model_id="Qwen/Qwen3-8B"))
    assert failing_ids(results) == {"model_id_matches_frozen"}


def test_branch_revision_fails_exact_revision_checks() -> None:
    results = evaluate_phase0(replace(valid_facts(), requested_revision="main"))
    assert {"requested_revision_is_commit", "resolved_revision_matches_requested"} <= (
        failing_ids(results)
    )


def test_thinking_mode_render_fails_template_check() -> None:
    thinking_render = (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )
    results = evaluate_phase0(replace(valid_facts(), rendered_prompt=thinking_render))
    assert "chat_template_non_thinking" in failing_ids(results)


def test_think_tag_leak_fails_response_check() -> None:
    results = evaluate_phase0(replace(valid_facts(), response_text="<think>x</think>ok"))
    assert "response_has_no_think_tags" in failing_ids(results)


def test_empty_response_fails_non_empty_check() -> None:
    results = evaluate_phase0(replace(valid_facts(), response_text="   "))
    assert "response_non_empty" in failing_ids(results)


def test_remote_code_module_fails_native_class_checks() -> None:
    results = evaluate_phase0(
        replace(valid_facts(), model_module="transformers_modules.Qwen3.4B.modeling_qwen3")
    )
    assert "model_native_no_remote_code" in failing_ids(results)


def test_wrong_dtype_or_device_fails_substrate_checks() -> None:
    results = evaluate_phase0(replace(valid_facts(), dtype_name="torch.float16"))
    assert "dtype_is_bfloat16" in failing_ids(results)

    results = evaluate_phase0(replace(valid_facts(), parameter_device="cpu"))
    assert "model_on_reference_device" in failing_ids(results)


def test_verdict_fail_when_any_check_fails() -> None:
    results = evaluate_phase0(replace(valid_facts(), cuda_available=False))
    assert verdict(results) == "FAIL"


def test_is_non_thinking_render_requires_closed_think_tail() -> None:
    assert is_non_thinking_render(NON_THINKING_RENDER)
    assert not is_non_thinking_render("<|im_start|>assistant\n")
    assert not is_non_thinking_render("<think>\nreasoning</think>\nanswer")
    assert not is_non_thinking_render("plain text without any think block")


def test_contains_think_tags_detects_both_tags() -> None:
    assert contains_think_tags("<think>")
    assert contains_think_tags("</think>")
    assert not contains_think_tags("READY")


def test_check_results_and_facts_serialize_to_json() -> None:
    results = evaluate_phase0(valid_facts())
    payload = [result.as_dict() for result in results]
    encoded = json.dumps(payload, sort_keys=True)
    assert "chat_template_non_thinking" in encoded

    facts = valid_facts()
    assert json.dumps(facts.rendered_prompt)
