from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from resonance_asri.contracts import ExecutionReceipt


class JsonlReceiptLedger:
    """Append-only local receipt ledger for experiment runs."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, receipt: ExecutionReceipt) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(receipt)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
