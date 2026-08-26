import numpy as np
from PIL import Image

from core.visualization import save_annotated_image


def test_save_annotated_image_draws_mask_contour_without_numbered_boxes(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "annotated.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(source)
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:7, 1:7] = True

    save_annotated_image(source, [], output, mask=mask)

    result = np.asarray(Image.open(output).convert("RGB"))
    assert np.count_nonzero(result[:, :, 0] > 200) > 0
    assert np.all(result[3:5, 3:5] == 0)


def test_save_annotated_image_supports_fluorescent_green(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "green.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(source)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    save_annotated_image(
        source,
        [],
        output,
        mask=mask,
        contour_color="#39FF14",
        contour_thickness=2,
    )

    result = np.asarray(Image.open(output).convert("RGB"))
    green_pixels = (result[:, :, 1] > 240) & (result[:, :, 0] < 100)
    assert np.count_nonzero(green_pixels) > 0


def test_save_annotated_image_supports_mask_fill_mode(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "mask_overlay.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(source)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    save_annotated_image(
        source,
        [],
        output,
        mask=mask,
        annotation_mode="mask",
        contour_color="#39FF14",
        mask_alpha=180,
    )

    result = np.asarray(Image.open(output).convert("RGB"))
    assert tuple(result[3, 3]) == (40, 180, 14)
    assert tuple(result[0, 0]) == (0, 0, 0)
