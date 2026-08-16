"""S0-B calibration: frozen task set, deterministic evaluators, paired deltas."""

from resonance_asri.calibration.analysis import (
    classify_population,
    paired_deltas,
    summarize_by_class,
    summarize_populations,
)
from resonance_asri.calibration.evaluators import (
    extract_numbers,
    normalize_text,
    score_response,
)
from resonance_asri.calibration.tasks import (
    CALIBRATION_SET_ID,
    CALIBRATION_TASKS,
    CalibrationTask,
    build_manifest,
    write_manifest,
)

__all__ = [
    "CALIBRATION_SET_ID",
    "CALIBRATION_TASKS",
    "CalibrationTask",
    "build_manifest",
    "classify_population",
    "extract_numbers",
    "normalize_text",
    "paired_deltas",
    "score_response",
    "summarize_by_class",
    "summarize_populations",
    "write_manifest",
]
