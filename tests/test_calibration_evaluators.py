from resonance_asri.calibration.evaluators import (
    extract_numbers,
    normalize_text,
    score_response,
)
from resonance_asri.calibration.tasks import CalibrationTask


def task(evaluator: str, expected) -> CalibrationTask:
    return CalibrationTask(
        task_id="t",
        task_class="c",
        difficulty="easy",
        prompt="p",
        evaluator=evaluator,
        expected=expected,
    )


def test_numeric_evaluator_matches_number_among_text() -> None:
    item = task("numeric", 66)
    assert score_response("The final price is $66.", item) == 1.0
    assert score_response("The final price is 65 dollars.", item) == 0.0
    assert score_response("80 - 25% + tax gives 66", item) == 1.0


def test_numeric_evaluator_handles_floats_and_negatives() -> None:
    item = task("numeric", 2.5)
    assert score_response("answer: 2.5", item) == 1.0
    item = task("numeric", -7)
    assert score_response("result = -7 units", item) == 1.0


def test_extract_numbers_finds_all() -> None:
    assert extract_numbers("3 apples, 2.5 kg, -4 degrees") == [3.0, 2.5, -4.0]


def test_exact_evaluator_normalizes_case_punctuation_and_spacing() -> None:
    item = task("exact", "Canberra")
    assert score_response("Canberra", item) == 1.0
    assert score_response("  canberra. ", item) == 1.0
    assert score_response("canberraa", item) == 0.0
    assert score_response("The capital is Canberra", item) == 0.0  # strict, by design


def test_option_evaluator_accepts_list_of_answers() -> None:
    item = task("option", ["no"])
    assert score_response("NO", item) == 1.0
    assert score_response("No.", item) == 1.0
    assert score_response("yes", item) == 0.0


def test_keyword_evaluator_scores_fraction_present() -> None:
    item = task("keywords", ["red", "blue", "yellow"])
    assert score_response("red, blue, yellow", item) == 1.0
    assert score_response("red and blue", item) == 2 / 3
    assert score_response("green", item) == 0.0


def test_normalize_text_strips_quotes_and_collapses_whitespace() -> None:
    assert normalize_text("  'Hello,   World!' ") == "hello world"


def test_unknown_evaluator_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        score_response("x", task("vibes", "y"))
