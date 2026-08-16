import pytest

from resonance_asri.reference.config import (
    DEFAULT_HF_HOME,
    FROZEN_QWEN_REFERENCE,
    dtype_label,
    hub_repo_dir_name,
    is_commit_revision,
)


def test_frozen_reference_pins_exact_identity() -> None:
    assert FROZEN_QWEN_REFERENCE.model_id == "Qwen/Qwen3-4B"
    assert FROZEN_QWEN_REFERENCE.revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert FROZEN_QWEN_REFERENCE.dtype_name == "bfloat16"
    assert FROZEN_QWEN_REFERENCE.device == "cuda:0"
    assert FROZEN_QWEN_REFERENCE.trust_remote_code is False
    assert FROZEN_QWEN_REFERENCE.enable_thinking is False


def test_frozen_revision_is_a_full_commit_sha() -> None:
    assert is_commit_revision(FROZEN_QWEN_REFERENCE.revision)


def test_is_commit_revision_rejects_non_sha_values() -> None:
    assert not is_commit_revision("main")
    assert not is_commit_revision("1cfa9a7")
    assert not is_commit_revision("1CFA9A7208912126459214E8B04321603B3DF60C")
    assert not is_commit_revision("z" * 40)


def test_snapshot_path_follows_hf_cache_layout() -> None:
    snapshot = FROZEN_QWEN_REFERENCE.snapshot_path
    assert snapshot == (
        DEFAULT_HF_HOME
        / "hub"
        / "models--Qwen--Qwen3-4B"
        / "snapshots"
        / FROZEN_QWEN_REFERENCE.revision
    )


def test_hub_repo_dir_name_replaces_slashes() -> None:
    assert hub_repo_dir_name("Qwen/Qwen3-4B") == "models--Qwen--Qwen3-4B"


def test_dtype_label_maps_known_names() -> None:
    assert dtype_label("bfloat16") == "torch.bfloat16"
    assert dtype_label("float16") == "torch.float16"


def test_dtype_label_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        dtype_label("int4")


def test_public_dict_omits_local_paths() -> None:
    public = FROZEN_QWEN_REFERENCE.as_public_dict()
    assert "hf_home" not in public
    assert "snapshot_path" not in public
    assert public["model_id"] == "Qwen/Qwen3-4B"
    assert public["dtype_label"] == "torch.bfloat16"
