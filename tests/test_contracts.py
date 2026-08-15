from resonance_asri import ExecutionReceipt, ExecutionRequest, ResourceUsage


def test_resource_usage_rejects_negative_values() -> None:
    try:
        ResourceUsage(provider_calls=-1)
    except ValueError as exc:
        assert "provider_calls" in str(exc)
    else:
        raise AssertionError("negative resource usage must be rejected")


def test_execution_receipt_contains_world_safe_measurements() -> None:
    request = ExecutionRequest(request_id="r1", prompt="Solve 2 + 2")
    usage = ResourceUsage(
        input_tokens=4,
        output_tokens=1,
        provider_calls=1,
        reasoning_iterations=1,
        latency_ms=10.0,
    )
    receipt = ExecutionReceipt(
        run_id="run-1",
        request_id=request.request_id,
        policy_id="fixed-shallow-v0",
        model_id="test-model",
        answer="4",
        usage=usage,
    )

    assert receipt.request_id == "r1"
    assert receipt.usage.provider_calls == 1
    assert receipt.specialists_activated == ()
