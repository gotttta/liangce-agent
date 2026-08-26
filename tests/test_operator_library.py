import pytest

from core.operator_library import OperatorLibrary


SPEC = {
    "name": "texture_mask",
    "input_artifact": "ImageArtifact",
    "output_artifact": "MaskArtifact",
    "atomic": True,
    "description": "亮度纹理阈值",
    "source": "def apply(data, params):\n    return data > np.mean(data)",
}


def test_operator_library_rejects_unreviewed_stage(tmp_path):
    library = OperatorLibrary(tmp_path / "operators")

    with pytest.raises(PermissionError, match="user-tested approval"):
        library.publish(SPEC, source_task_id="task_1")

    assert library.list_operators() == []


def test_operator_library_publishes_and_reloads_atomic_stage(tmp_path):
    library = OperatorLibrary(tmp_path / "operators")
    published = library.publish(
        SPEC,
        source_task_id="task_1",
        user_tested=True,
        tested_by="user",
        test_note="manual sample test passed",
    )

    loaded = library.get("texture_mask")

    assert loaded["id"] == published["id"]
    assert loaded["source_task_id"] == "task_1"
    assert loaded["atomic"] is True
    assert loaded["approval"]["user_tested"] is True
    assert loaded["approval"]["tested_by"] == "user"
