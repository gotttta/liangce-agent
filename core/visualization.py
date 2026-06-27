import numpy as np
from PIL import Image, ImageDraw


def save_mask_image(mask, output_path):
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image.save(output_path)


def save_annotated_image(source_path, results, output_path):
    image = Image.open(source_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for item in results:
        x, y, width, height = item["bbox"]
        draw.rectangle(
            [x, y, x + width - 1, y + height - 1],
            outline=(255, 48, 48, 255),
            width=2,
        )
        draw.text((x, max(0, y - 12)), str(item["id"]), fill=(255, 48, 48, 255))

    annotated = Image.alpha_composite(image.convert("RGBA"), overlay)
    annotated.convert("RGB").save(output_path)
