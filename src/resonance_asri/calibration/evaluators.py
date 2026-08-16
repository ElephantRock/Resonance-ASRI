"""Deterministic calibration evaluators.

No LLM judging anywhere: quality is a pure function of (response text, task).
Binary scores (0.0 / 1.0) for exact/numeric/option; the keyword evaluator is
graded (fraction of required keywords present). The same evaluator runs for
every condition, so scoring cannot favor one condition over another.
"""

from __future__ import annotations

import re
from typing import Any

from resonance_asri.calibration.tasks import CalibrationTask

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_PUNCTUATION_PATTERN = re.compile(r"[.,!?;:\"'()\[\]{}]+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation/quotes, collapse whitespace."""

    stripped = _PUNCTUATION_PATTERN.sub(" ", text.casefold())
    return " ".join(stripped.split()).strip()


def extract_numbers(text: str) -> list[float]:
    """All numbers appearing in the response, as floats."""

    return [float(match) for match in _NUMBER_PATTERN.findall(text)]


def score_response(response: str, task: CalibrationTask) -> float:
    """Score one response against its task; returns a value in [0, 1]."""

    evaluator = task.evaluator
    expected = task.expected

    if evaluator == "numeric":
        numbers = extract_numbers(response)
        target = float(expected)
        return 1.0 if any(abs(number - target) < 1e-6 for number in numbers) else 0.0

    if evaluator == "exact":
        return 1.0 if normalize_text(response) == normalize_text(str(expected)) else 0.0

    if evaluator == "option":
        normalized = normalize_text(response)
        acceptable = {normalize_text(str(option)) for option in expected}
        return 1.0 if normalized in acceptable else 0.0

    if evaluator == "keywords":
        normalized = normalize_text(response)
        required = [normalize_text(str(keyword)) for keyword in expected]
        present = sum(1 for keyword in required if keyword in normalized)
        return present / len(required) if required else 0.0

    raise ValueError(f"unknown evaluator: {evaluator!r}")


def evaluator_summary(task: CalibrationTask) -> dict[str, Any]:
    return {"task_id": task.task_id, "evaluator": task.evaluator, "expected": task.expected}
