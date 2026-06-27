from providers.vision import MockVisionProvider, extract_json_object


def test_mock_provider_returns_normalized_strategy():
    provider = MockVisionProvider()

    strategy = provider.create_strategy(
        target_image_path="target.png",
        description="找亮色残留，量面积和数量",
        reference_annotation_path=None,
        previous_state=None,
    )

    assert strategy["measurement_type"] == "area_count"
    assert strategy["segmentation"]["method"] == "bright_threshold"
    assert strategy["segmentation"]["min_area_px"] == 20


def test_extract_json_object_from_markdown_response():
    text = '观察如下：```json\n{"defect_type": "residue", "confidence": 0.8}\n```'

    parsed = extract_json_object(text)

    assert parsed == {"defect_type": "residue", "confidence": 0.8}
