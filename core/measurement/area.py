from collections import deque

import numpy as np


def measure_components(mask, min_area=20, unit="pixel"):
    visited = np.zeros(mask.shape, dtype=bool)
    results = []
    component_id = 1

    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            pixels = _collect_component(mask, visited, x, y)
            area = len(pixels)
            if area < min_area:
                continue

            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox_width = max_x - min_x + 1
            bbox_height = max_y - min_y + 1
            aspect_ratio = bbox_width / bbox_height if bbox_height else 0.0

            results.append({
                "id": component_id,
                "area": area,
                "bbox": [min_x, min_y, bbox_width, bbox_height],
                "width": bbox_width,
                "height": bbox_height,
                "aspect_ratio": round(aspect_ratio, 4),
                "unit": unit,
                "confidence": 0.5,
            })
            component_id += 1

    total_area = int(sum(item["area"] for item in results))
    image_area = int(mask.size)
    area_ratio = round(total_area / image_area, 6) if image_area else 0.0
    return {
        "results": results,
        "summary": {
            "count": len(results),
            "total_area": total_area,
            "unit": unit,
            "area_ratio": area_ratio,
        },
    }


def _collect_component(mask, visited, start_x, start_y):
    queue = deque([(start_x, start_y)])
    visited[start_y, start_x] = True
    pixels = []

    while queue:
        x, y = queue.popleft()
        pixels.append((x, y))
        for nx, ny in _neighbors(x, y, mask.shape[1], mask.shape[0]):
            if visited[ny, nx] or not mask[ny, nx]:
                continue
            visited[ny, nx] = True
            queue.append((nx, ny))

    return pixels


def _neighbors(x, y, width, height):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny
