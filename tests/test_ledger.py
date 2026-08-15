from __future__ import annotations

import json

from resonance_asri.contracts import ExecutionReceipt, ResourceUsage
from resonance_asri.telemetry import JsonlReceiptLedger


def test_jsonl_ledger_appends_receipts(tmp_path) -> None:
    path = tmp_path / "receipts.jsonl"
    ledger = JsonlReceiptLedger(path)
    receipt = ExecutionReceipt(
        run_id="run-1",
        request_id="request-1",
        policy_id="fixed-shallow-v0",
        model_id="fake-model",
        answer="4",
        usage=ResourceUsage(provider_calls=1, reasoning_iterations=1),
    )

    ledger.append(receipt)

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["run_id"] == "run-1"
    assert payload["usage"]["provider_calls"] == 1
