"""Frozen heterogeneous S0-B calibration task set.

Thirty-six tasks across six classes with a difficulty spread inside each
class. Every task has a deterministic evaluator (no LLM judging), so quality
scores are exactly reproducible. Task prompts are deliberately short: with the
runtime's wrapper text every rendered prompt stays far inside the frozen
envelope (<= 1024 prompt tokens, <= 256 generated tokens per call).

This module is the frozen manifest source: the runner emits it as
``manifest.json`` (with a sha256) before any condition runs, and calibration
data must reference that hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

CALIBRATION_SET_ID = "s0b-calibration-v1"


@dataclass(frozen=True, slots=True)
class CalibrationTask:
    """One deterministic calibration task."""

    task_id: str
    task_class: str
    difficulty: str
    prompt: str
    evaluator: str  # "exact" | "numeric" | "option" | "keywords"
    expected: Any  # str for exact, number for numeric, list[str] for option/keywords

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _t(
    task_id: str,
    task_class: str,
    difficulty: str,
    prompt: str,
    evaluator: str,
    expected: Any,
) -> CalibrationTask:
    return CalibrationTask(
        task_id=task_id,
        task_class=task_class,
        difficulty=difficulty,
        prompt=prompt,
        evaluator=evaluator,
        expected=expected,
    )


CALIBRATION_TASKS: tuple[CalibrationTask, ...] = (
    # --- arithmetic (numeric evaluator) -----------------------------------
    _t("arith-01", "arithmetic", "easy", "Compute 17 + 26. Answer with only the number.", "numeric", 43),
    _t("arith-02", "arithmetic", "easy", "Compute 12 x 12. Answer with only the number.", "numeric", 144),
    _t("arith-03", "arithmetic", "medium", "Compute 1024 - 387. Answer with only the number.", "numeric", 637),
    _t("arith-04", "arithmetic", "medium", "Compute 144 / 16 + 27. Answer with only the number.", "numeric", 36),
    _t("arith-05", "arithmetic", "hard", "Compute 138 x 46. Answer with only the number.", "numeric", 6348),
    _t("arith-06", "arithmetic", "hard", "Compute 19 x 23 x 7. Answer with only the number.", "numeric", 3059),
    # --- word_problem (numeric evaluator) ----------------------------------
    _t(
        "word-01", "word_problem", "easy",
        "A train travels 240 km in 3 hours at constant speed. What is its average "
        "speed in km/h? Answer with only the number.",
        "numeric", 80,
    ),
    _t(
        "word-02", "word_problem", "easy",
        "A shop sells apples at $2 each and oranges at $3 each. Maya buys 3 apples "
        "and 2 oranges. What is the total cost in dollars? Answer with only the number.",
        "numeric", 12,
    ),
    _t(
        "word-03", "word_problem", "medium",
        "A jacket costs $80. A 25% discount is applied, then 10% tax is added to the "
        "discounted price. What is the final price in dollars? Answer with only the number.",
        "numeric", 66,
    ),
    _t(
        "word-04", "word_problem", "medium",
        "A tank holds 96 liters. One pipe fills it at 12 liters per minute and a drain "
        "empties it at 4 liters per minute. With both open, how many minutes does the "
        "tank take to fill? Answer with only the number.",
        "numeric", 12,
    ),
    _t(
        "word-05", "word_problem", "hard",
        "Worker A finishes a job in 6 hours and worker B finishes the same job in 3 "
        "hours. Working together at constant rates, how many hours do they need? "
        "Answer with only the number.",
        "numeric", 2,
    ),
    _t(
        "word-06", "word_problem", "hard",
        "An arithmetic sequence starts 7, 12, 17, ... with a constant step. What is "
        "the 20th term? Answer with only the number.",
        "numeric", 102,
    ),
    # --- code_reasoning (exact/numeric evaluator on printed output) --------
    _t(
        "code-01", "code_reasoning", "easy",
        "What does this Python code print?\n\nx = 0\nfor i in range(4):\n    x += i\nprint(x)\n\n"
        "Answer with only the printed value.",
        "numeric", 6,
    ),
    _t(
        "code-02", "code_reasoning", "easy",
        "What does this Python code print?\n\ns = 'engines'\nprint(len(s))\n\n"
        "Answer with only the printed value.",
        "numeric", 7,
    ),
    _t(
        "code-03", "code_reasoning", "medium",
        "What does this Python code print?\n\nd = {x: x * x for x in (2, 3, 5)}\n"
        "print(d[3] + d[5])\n\nAnswer with only the printed value.",
        "numeric", 34,
    ),
    _t(
        "code-04", "code_reasoning", "medium",
        "What does this Python code print?\n\na = [1, 2, 3]\na.append(4)\na.pop(0)\n"
        "print(len(a) + a[0])\n\nAnswer with only the printed value.",
        "numeric", 5,
    ),
    _t(
        "code-05", "code_reasoning", "hard",
        "What does this Python code print?\n\ndef f(n):\n    return n if n < 2 else "
        "f(n - 1) + f(n - 2)\nprint(f(7))\n\nAnswer with only the printed value.",
        "numeric", 13,
    ),
    _t(
        "code-06", "code_reasoning", "hard",
        "What does this Python code print?\n\ns = 'resonance'\nprint(s[3:7])\n\n"
        "Answer with only the printed value.",
        "exact", "onan",
    ),
    # --- lookup (exact/option evaluator on stable facts) -------------------
    _t("fact-01", "lookup", "easy", "What is the capital city of Australia? Answer with only the city name.", "exact", "Canberra"),
    _t("fact-02", "lookup", "easy", "What is the chemical symbol for gold? Answer with only the symbol.", "exact", "Au"),
    _t("fact-03", "lookup", "easy", "How many days does a leap year have? Answer with only the number.", "numeric", 366),
    _t("fact-04", "lookup", "easy", "Which is the largest planet in the solar system? Answer with only the planet name.", "exact", "Jupiter"),
    _t("fact-05", "lookup", "medium", "In which year did the Berlin Wall fall? Answer with only the year.", "numeric", 1989),
    _t("fact-06", "lookup", "medium", "How many bones are in the adult human body? Answer with only the number.", "numeric", 206),
    # --- reasoning (option/exact evaluator, multi-step chains) -------------
    _t(
        "logic-01", "reasoning", "easy",
        "Alice is taller than Bob. Carol is shorter than Bob. Who is the tallest of "
        "the three? Answer with only the name.",
        "exact", "Alice",
    ),
    _t(
        "logic-02", "reasoning", "easy",
        "If today is Wednesday, what day of the week will it be exactly 100 days "
        "from now? Answer with only the day name.",
        "exact", "Friday",
    ),
    _t(
        "logic-03", "reasoning", "medium",
        "All Blorks are Zibs. Some Zibs are Quins. Can we conclude that some Blorks "
        "are definitely Quins? Answer YES or NO only.",
        "option", ["no"],
    ),
    _t(
        "logic-04", "reasoning", "medium",
        "When two fair six-sided dice are rolled, what is the probability that the "
        "two numbers sum to 7? Answer with only the fraction, like 1/2.",
        "exact", "1/6",
    ),
    _t(
        "logic-05", "reasoning", "hard",
        "A cube is painted red on all sides and then cut into 27 equal small cubes. "
        "How many small cubes have exactly two red faces? Answer with only the number.",
        "numeric", 12,
    ),
    _t(
        "logic-06", "reasoning", "hard",
        "Five people sit in a row. Person A is at the far left. Person B is directly "
        "right of A. Person C is at the far right. Person D sits between B and "
        "person E, with D closer to B. Who is in the middle seat of the five? "
        "Answer with only the name.",
        "exact", "D",
    ),
    # --- instruction_following (exact/option/keywords on compliance) -------
    _t("instr-01", "instruction_following", "easy", "Reply with exactly the word BANANA and nothing else.", "exact", "BANANA"),
    _t("instr-02", "instruction_following", "easy", "Answer with a single digit only: what is 3 + 4?", "exact", "7"),
    _t(
        "instr-03", "instruction_following", "easy",
        "Answer YES or NO only: is 51 a prime number?",
        "option", ["no"],
    ),
    _t("instr-04", "instruction_following", "medium", "How many letters are in the word 'resonance'? Answer with only the number.", "numeric", 9),
    _t("instr-05", "instruction_following", "medium", "What is the two-letter postal abbreviation for the US state of California? Answer with only the abbreviation.", "exact", "CA"),
    _t(
        "instr-06", "instruction_following", "medium",
        "List exactly the three primary colors of pigment, separated by commas.",
        "keywords", ["red", "blue", "yellow"],
    ),
)


def build_manifest() -> dict[str, Any]:
    """Frozen manifest payload with a content hash over the task list."""

    tasks = [task.as_dict() for task in CALIBRATION_TASKS]
    digest = hashlib.sha256(
        json.dumps(tasks, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "calibration_set_id": CALIBRATION_SET_ID,
        "task_count": len(tasks),
        "tasks_sha256": digest,
        "evaluators": ["exact", "numeric", "option", "keywords"],
        "classes": sorted({task["task_class"] for task in tasks}),
        "tasks": tasks,
    }


def write_manifest(path: str) -> dict[str, Any]:
    """Write the manifest JSON and return the payload."""

    from pathlib import Path

    payload = build_manifest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload
