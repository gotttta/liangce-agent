import json
from pathlib import Path

import pytest

from core.pipelines.dsl import pipeline_operator_catalog, validate_pipeline
from core.skills import SkillRegistry, skill_catalog


def legacy_skill():
    return json.loads((Path(__file__).parent / "fixtures" / "legacy_skill.json").read_text())


@pytest.fixture
def workspace_skill(tmp_path):
    directory = tmp_path / 'custom'
    directory.mkdir()
    (directory / 'SKILL.md').write_text(
        '---\nname: custom\ndescription: custom measurement\n---\nBODY', encoding='utf-8')
    references = directory / 'references'
    references.mkdir()
    (references / 'details.md').write_text('OPTIONAL DETAILS', encoding='utf-8')
    return directory


def test_catalog_contains_exactly_six_measurement_skills():
    registry = SkillRegistry()
    expected = {"line_width_spacing", "hole_diameter", "position_offset",
                "area_measurement", "contour_deviation", "defect_count"}
    assert {item["name"] for item in registry.list_skills()} == expected
    for name in expected:
        skill = registry.get(name)
        assert 'content' in skill and 'pipeline_template' not in skill
        assert skill['resources'] == []
    for removed in ("generic_bright_blob_detection", "generic_dark_blob_detection", "periodic_particle_detection"):
        assert registry.get(removed) is None


def test_operator_catalog_is_registry_backed_and_has_v3_port_contracts():
    catalog = {item["name"]: item for item in pipeline_operator_catalog()}

    assert catalog["build_periodic_background"]["version"] == "1.0.0"
    assert catalog["build_periodic_background"]["input_ports"] == {
        "image": "ImageArtifact",
        "period": "MetadataArtifact",
    }
    assert "periodic_background_model" not in catalog
    assert any(item["name"] == "defect_count" for item in skill_catalog())


def test_accepted_legacy_workspace_skill_loads_with_operator_vocabulary(tmp_path):
    import json
    from core.tools.discovery import dispatch_discovery

    skill = legacy_skill()
    skill["name"] = "saved_legacy_skill"
    skill["status"] = "accepted"
    skill["required_tools"] = skill.pop("required_operators")
    for node in skill["pipeline_template"]["nodes"]:
        node["tool"] = node.pop("operator")
    directory = tmp_path / "skill_legacy"
    directory.mkdir()
    (directory / "skill.json").write_text(json.dumps(skill))
    result, _ = dispatch_discovery({"tool": "load_skill", "arguments": {"name": skill["name"]}}, {}, tmp_path)
    loaded = result["skill"]
    assert "required_tools" not in loaded
    assert set(loaded["required_operators"]) == {item["name"] for item in result["operators"]}
    assert all("tool" not in node and "operator" in node for node in loaded["pipeline_template"]["nodes"])
    validate_pipeline(loaded["pipeline_template"])


def test_workspace_schema_versions_and_cache_invalidation(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import pytest

    registry = SkillRegistry(tmp_path)
    old = legacy_skill()
    old.update(name='workspace_blob', status='accepted')
    directory = tmp_path / 'skill_old'
    directory.mkdir()
    path = directory / 'skill.json'
    path.write_text(json.dumps(old))
    newer = {**old, 'version': '1.2.0'}
    second = tmp_path / 'skill_new'
    second.mkdir()
    (second / 'skill.json').write_text(json.dumps(newer))
    reads = []
    original_read = Path.read_text
    def read(self, *args, **kwargs):
        reads.append(self)
        return original_read(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    assert registry.get('workspace_blob')['version'] == '1.2.0'
    assert registry.get('workspace_blob', version='1.0.0')['version'] == '1.0.0'
    assert len(reads) == 2
    newer['version'] = '2.0.0'
    (second / 'skill.json').write_text(json.dumps(newer))
    assert registry.get('workspace_blob')['version'] == '2.0.0'
    assert len(reads) == 3
    old['required_operators'] = []
    path.write_text(json.dumps(old))
    with pytest.warns(RuntimeWarning, match='dependencies'):
        assert registry.get('workspace_blob', version='1.0.0') is None
    assert registry.errors


def test_duplicate_builtin_version_cannot_override_and_invalid_file_is_isolated(tmp_path):
    import json
    import pytest
    registry = SkillRegistry(tmp_path)
    skill = legacy_skill()
    skill['status'] = 'accepted'
    for name, content in [('duplicate', json.dumps(skill)), ('invalid', '[]')]:
        directory = tmp_path / f'skill_{name}'
        directory.mkdir()
        (directory / 'skill.json').write_text(content)
    with pytest.warns(RuntimeWarning):
        loaded = registry.list_skills()
    assert len(loaded) == 6
    assert len(registry.errors) == 2


def test_progressive_disclosure_and_resource_boundaries(tmp_path, workspace_skill, monkeypatch):
    from core.tools.discovery import dispatch_discovery
    assert all(set(item) == {'name', 'description'} for item in skill_catalog())
    reads = []
    original_read = Path.read_text
    def read(self, *args, **kwargs):
        reads.append(self)
        return original_read(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    registry = SkillRegistry(tmp_path)
    result, _ = dispatch_discovery({'tool': 'load_skill', 'arguments': {'name': 'custom'}}, {}, tmp_path, registry)
    assert result['skill']['content'].endswith('BODY')
    assert result['skill']['resources'] == ['references/details.md']
    assert 'OPTIONAL DETAILS' not in result['skill']['content']
    assert reads == [(workspace_skill / 'SKILL.md').resolve()]
    result, _ = dispatch_discovery({'tool': 'load_skill', 'arguments': {
        'name': 'custom', 'resource': 'references/details.md'}}, {}, tmp_path, registry)
    assert result['resource']['content'] == 'OPTIONAL DETAILS'
    assert reads[-1] == (workspace_skill / 'references/details.md').resolve()
    for path in ('../../README.md', '/etc/passwd'):
        with pytest.raises(ValueError):
            registry.read_resource('custom', path)
    (workspace_skill / 'escape.md').symlink_to(tmp_path / 'secret.md')
    (tmp_path / 'secret.md').write_text('secret')
    assert 'escape.md' not in registry.get('custom')['resources']
    with pytest.raises(ValueError):
        registry.read_resource('custom', 'escape.md')


def test_catalog_does_not_read_skill_body_or_references(tmp_path, workspace_skill, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('catalog must not read full documents')
    monkeypatch.setattr(Path, 'read_text', forbidden)
    catalog = skill_catalog(tmp_path)
    assert len(catalog) == 7
    assert {'name': 'custom', 'description': 'custom measurement'} in catalog
