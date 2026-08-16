"""Frozen-reference tooling for ASRI Phase-0 validation.

Only torch-free pieces are re-exported here so importing this package (and the
unit tests that cover it) never requires a GPU environment. The loader lives in
``resonance_asri.reference.loader`` and is imported only by GPU runners.
"""

from resonance_asri.reference.config import (
    DEFAULT_HF_HOME,
    FROZEN_QWEN_REFERENCE,
    QwenReferenceConfig,
    dtype_label,
    is_commit_revision,
)
from resonance_asri.reference.validation import (
    CheckResult,
    Phase0Facts,
    evaluate_phase0,
    verdict,
)

__all__ = [
    "DEFAULT_HF_HOME",
    "FROZEN_QWEN_REFERENCE",
    "CheckResult",
    "Phase0Facts",
    "QwenReferenceConfig",
    "dtype_label",
    "evaluate_phase0",
    "is_commit_revision",
    "verdict",
]
