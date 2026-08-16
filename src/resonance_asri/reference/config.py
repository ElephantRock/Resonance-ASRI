"""Frozen Qwen3-4B reference configuration for ASRI Phase-0 validation.

The values in this module are experimental constants. They identify the exact
substrate every S0 measurement must run on and must not drift between runs.
The module is intentionally dependency-free so unit tests and CI never need
torch, transformers, or a GPU to import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

QWEN3_4B_MODEL_ID = "Qwen/Qwen3-4B"
QWEN3_4B_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
REFERENCE_DTYPE_NAME = "bfloat16"
REFERENCE_DEVICE = "cuda:0"
DEFAULT_HF_HOME = Path(r"C:\huggingface_cache")

SMOKE_PROMPT = "Return exactly the word READY."
SMOKE_MAX_NEW_TOKENS = 16
EXPECTED_SMOKE_RESPONSE = "READY"

_DTYPE_LABELS = {
    "bfloat16": "torch.bfloat16",
    "float16": "torch.float16",
    "float32": "torch.float32",
}


def is_commit_revision(value: str) -> bool:
    """Return True when ``value`` is a full 40-character lowercase git SHA."""

    return len(value) == 40 and all(char in "0123456789abcdef" for char in value)


def hub_repo_dir_name(model_id: str) -> str:
    """Map a hub model id to its local cache directory name."""

    return "models--" + model_id.replace("/", "--")


def dtype_label(dtype_name: str) -> str:
    """Return the torch-facing label for a dtype name (e.g. ``torch.bfloat16``)."""

    try:
        return _DTYPE_LABELS[dtype_name]
    except KeyError as exc:
        raise ValueError(f"unsupported reference dtype: {dtype_name!r}") from exc


@dataclass(frozen=True, slots=True)
class QwenReferenceConfig:
    """Frozen identity of the local Qwen reference substrate."""

    model_id: str
    revision: str
    dtype_name: str
    device: str
    trust_remote_code: bool
    enable_thinking: bool
    hf_home: Path

    @property
    def hub_dir(self) -> Path:
        return self.hf_home / "hub"

    @property
    def snapshot_path(self) -> Path:
        return self.hub_dir / hub_repo_dir_name(self.model_id) / "snapshots" / self.revision

    def as_public_dict(self) -> dict[str, object]:
        """JSON-ready identity without machine-local absolute paths."""

        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype_name": self.dtype_name,
            "dtype_label": dtype_label(self.dtype_name),
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
            "enable_thinking": self.enable_thinking,
        }


FROZEN_QWEN_REFERENCE = QwenReferenceConfig(
    model_id=QWEN3_4B_MODEL_ID,
    revision=QWEN3_4B_REVISION,
    dtype_name=REFERENCE_DTYPE_NAME,
    device=REFERENCE_DEVICE,
    trust_remote_code=False,
    enable_thinking=False,
    hf_home=DEFAULT_HF_HOME,
)
