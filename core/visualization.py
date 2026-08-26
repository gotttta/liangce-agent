import numpy as np
from PIL import Image, ImageDraw


def save_mask_image(mask, output_path):
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image.save(output_path)


def save_annotated_image(
    source_path,
    results,
    output_path,
    mask=None,
    contour_color="#ff4030",
    contour_thickness=1,
    annotation_mode="contour",
    mask_alpha=72,
):
    """Render a task result without coupling the algorithm to one display style."""
    color = _parse_color(contour_color)
    if not isinstance(contour_thickness, int) or not 1 <= contour_thickness <= 10:
        raise ValueError("contour_thickness must be an integer between 1 and 10")
    image = Image.open(source_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if annotation_mode == "mask" and mask is not None:
        alpha = max(0, min(255, int(mask_alpha)))
        mask_array = np.asarray(mask, dtype=bool)
        fill = Image.new("RGBA", image.size, (*color, alpha))
        mask_layer = Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L")
        overlay = Image.composite(fill, overlay, mask_layer)
        annotated = Image.alpha_composite(image.convert("RGBA"), overlay)
        annotated.convert("RGB").save(output_path)
        return

    if annotation_mode == "bbox":
        for item in results or []:
            x, y, width, height = item["bbox"]
            draw.rectangle(
                [x, y, x + width - 1, y + height - 1],
                outline=(*color, 255),
                width=contour_thickness,
            )
            draw.text((x, max(0, y - 12)), str(item["id"]), fill=(*color, 255))
        annotated = Image.alpha_composite(image.convert("RGBA"), overlay)
        annotated.convert("RGB").save(output_path)
        return

    if mask is not None:
        for x, y in _mask_boundary(mask):
            radius = contour_thickness - 1
            if radius == 0:
                draw.point((x, y), fill=(*color, 255))
            else:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 255))
        annotated = Image.alpha_composite(image.convert("RGBA"), overlay)
        annotated.convert("RGB").save(output_path)
        return

    annotated = Image.alpha_composite(image.convert("RGBA"), overlay)
    annotated.convert("RGB").save(output_path)


def _parse_color(value):
    text = str(value or "").strip()
    if len(text) == 7 and text.startswith("#"):
        try:
            return tuple(int(text[index:index + 2], 16) for index in (1, 3, 5))
        except ValueError:
            pass
    raise ValueError("contour_color must be a #RRGGBB value")


def _mask_boundary(mask):
    """Return 4-connected boundary pixels without wrapping at image edges."""
    import numpy as np

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a 2D array")

    boundary = np.zeros_like(binary)
    boundary[1:, :] |= binary[1:, :] & ~binary[:-1, :]
    boundary[:-1, :] |= binary[:-1, :] & ~binary[1:, :]
    boundary[:, 1:] |= binary[:, 1:] & ~binary[:, :-1]
    boundary[:, :-1] |= binary[:, :-1] & ~binary[:, 1:]
    rows, columns = np.nonzero(boundary)
    return list(zip(columns, rows))
