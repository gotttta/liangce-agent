from pathlib import Path

from scripts.run_gate2_pipeline import _display_path, ROOT


def test_display_path_supports_external_output_directory(tmp_path):
    path = tmp_path / "sample" / "mask.png"
    assert _display_path(path) == str(path.resolve())


def test_display_path_uses_repo_relative_paths():
    path = ROOT / "outputs" / "gate2" / "mask.png"
    assert _display_path(path) == "outputs/gate2/mask.png"
