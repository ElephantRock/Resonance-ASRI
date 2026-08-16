from resonance_asri.calibration.tasks import (
    CALIBRATION_SET_ID,
    CALIBRATION_TASKS,
    build_manifest,
)

EXPECTED_CLASS_COUNTS = {
    "arithmetic": 6,
    "word_problem": 6,
    "code_reasoning": 6,
    "lookup": 6,
    "reasoning": 6,
    "instruction_following": 6,
}

VALID_EVALUATORS = {"exact", "numeric", "option", "keywords"}


def test_manifest_has_36_unique_tasks_across_six_classes() -> None:
    ids = [item.task_id for item in CALIBRATION_TASKS]
    assert len(ids) == 36
    assert len(set(ids)) == 36

    counts: dict[str, int] = {}
    for item in CALIBRATION_TASKS:
        counts[item.task_class] = counts.get(item.task_class, 0) + 1
    assert counts == EXPECTED_CLASS_COUNTS


def test_every_task_has_valid_evaluator_and_expected_value() -> None:
    for item in CALIBRATION_TASKS:
        assert item.evaluator in VALID_EVALUATORS, item.task_id
        assert item.prompt.strip(), item.task_id
        assert item.difficulty in {"easy", "medium", "hard"}, item.task_id
        if item.evaluator in {"option", "keywords"}:
            assert isinstance(item.expected, list) and item.expected, item.task_id
        else:
            assert item.expected not in (None, ""), item.task_id


def test_prompts_stay_well_inside_frozen_envelope() -> None:
    # 1024-token prompt ceiling; conservative bound of ~4 chars/token.
    for item in CALIBRATION_TASKS:
        assert len(item.prompt) <= 600, (item.task_id, len(item.prompt))


def test_difficulty_spread_exists_inside_each_class() -> None:
    by_class: dict[str, set[str]] = {}
    for item in CALIBRATION_TASKS:
        by_class.setdefault(item.task_class, set()).add(item.difficulty)
    assert all(len(levels) >= 2 for levels in by_class.values())


def test_manifest_payload_hash_is_stable_and_complete() -> None:
    manifest = build_manifest()
    assert manifest["calibration_set_id"] == CALIBRATION_SET_ID
    assert manifest["task_count"] == 36
    assert len(manifest["tasks_sha256"]) == 64
    assert sorted(manifest["classes"]) == sorted(EXPECTED_CLASS_COUNTS)

    again = build_manifest()
    assert again["tasks_sha256"] == manifest["tasks_sha256"]
