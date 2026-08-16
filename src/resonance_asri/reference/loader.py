"""Load the frozen Qwen3-4B reference substrate from the local HF cache.

This module imports torch and transformers at module level and is therefore
only imported by GPU runner scripts and the local provider, never by unit
tests or CI. Loading is offline and path-based: the pinned commit's snapshot
directory must already exist under ``hf_home``; nothing is downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from resonance_asri.reference.config import FROZEN_QWEN_REFERENCE, QwenReferenceConfig, dtype_label

_TORCH_DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}


class ReferenceLoadError(RuntimeError):
    """Raised when the frozen snapshot is missing or does not match its pin."""


@dataclass(frozen=True, slots=True)
class LoadedQwenReference:
    """Loaded tokenizer/model pair plus resolved identity."""

    tokenizer: object
    model: object
    device: torch.device
    snapshot_path: Path
    resolved_revision: str
    config: QwenReferenceConfig

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def dtype_name(self) -> str:
        return dtype_label(self.config.dtype_name)


def resolve_snapshot(config: QwenReferenceConfig = FROZEN_QWEN_REFERENCE) -> Path:
    """Return the pinned snapshot directory, failing loudly when absent."""

    snapshot = config.snapshot_path
    if not snapshot.is_dir():
        raise ReferenceLoadError(
            f"frozen snapshot not found at {snapshot}. The pinned revision must be "
            "present in the local Hugging Face cache before Phase-0 runs; no "
            "download is attempted."
        )
    return snapshot


def load_frozen_qwen_reference(
    config: QwenReferenceConfig = FROZEN_QWEN_REFERENCE,
) -> LoadedQwenReference:
    """Load tokenizer and model once, offline, under the frozen identity."""

    snapshot = resolve_snapshot(config)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=config.trust_remote_code)

    label = dtype_label(config.dtype_name)
    dtype = _TORCH_DTYPES[label]
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )

    if not torch.cuda.is_available():
        raise ReferenceLoadError(
            "CUDA is not available; the frozen reference substrate requires cuda:0"
        )

    device = torch.device(config.device)
    model.to(device)
    model.eval()

    return LoadedQwenReference(
        tokenizer=tokenizer,
        model=model,
        device=device,
        snapshot_path=snapshot,
        resolved_revision=snapshot.name,
        config=config,
    )
