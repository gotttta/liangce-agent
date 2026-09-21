"""Read exact, paginated evidence or aligned image crops from scoped experiments."""
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image

from core.experiments.context import artifact_view, bounded_value, json_size


REPORTS = {name: name + ".json" for name in (
    "measurements", "outputs", "quality_report", "operator_trace", "pipeline")}
PAGE_CHARS = 8000


def _report_preview(value):
    # Reports may nest outputs under quality or structured_outputs. Do not let
    # a small pixel array slip through merely because it fits the page budget.
    if isinstance(value, dict):
        if value.get("kind") in {"mask", "image", "contours", "metadata"} and "data" in value:
            return artifact_view(value)
        return {key: _report_preview(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_report_preview(item) for item in value]
    return value


def inspect_experiment(args, attempts, target, output_root):
    experiment_id = args.get("experiment_id")
    if experiment_id not in attempts:
        raise ValueError("experiment is outside this session")
    attempt = attempts[experiment_id]
    directory = Path(attempt["directory"])
    manifest = directory / "experiment.json"
    if manifest.exists():
        source_hash = json.loads(manifest.read_text(encoding="utf-8")).get("input_sha256")
        if source_hash and source_hash != sha256(Path(target).read_bytes()).hexdigest():
            raise ValueError("cannot inspect an experiment from a different input image")
    if "region" in args:
        if set(args) - {"experiment_id", "region"}:
            raise ValueError("region cannot be combined with report arguments")
        return _crop_evidence(experiment_id, directory, target, args["region"], output_root)
    report = args.get("report", "measurements")
    if report not in REPORTS:
        raise ValueError("unknown report")
    selector = args.get("selector", "")
    if selector and not selector.startswith("/"):
        raise ValueError("selector must be a JSON Pointer, e.g. /results or /measurements/data/components")
    value = json.loads((directory / REPORTS[report]).read_text(encoding="utf-8"))
    for token in selector.split("/")[1:]:
        # Raw pixels and complete contours are visual evidence, not text pages.
        if isinstance(value, dict) and value.get("kind") in {"image", "mask", "contours"} and token == "data":
            raise ValueError("pixel/contour arrays are not text evidence; request a region instead")
        key = token.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(value, list) and (not key.isdigit() or int(key) >= len(value)):
                raise ValueError("invalid list index")
            value = value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError("selector does not exist in this report") from None
    offset, limit = args.get("offset", 0), args.get("limit", 10)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    base = {"experiment_id": experiment_id, "report": report, "selector": selector,
            "offset": offset, "source_path": str(directory / REPORTS[report])}
    if isinstance(value, dict) and value.get("kind") in {"mask", "image", "contours", "metadata"}:
        return {**base, "summary": artifact_view(value),
                "note": "For metadata details select /data beneath this selector; pixels/contours use image inspection."}, []
    if not isinstance(value, (dict, list)):
        if not isinstance(value, str):
            return {**base, "value": value, "partial": False}, []
        # Code or long strings are retrieved exactly, in character pages.
        if offset > len(value):
            raise ValueError("offset exceeds text length")
        end = min(len(value), offset + PAGE_CHARS)
        return {**base, "value": value[offset:end], "total_chars": len(value),
                "next_offset": end if end < len(value) else None}, []
    items = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
    if offset > len(items):
        raise ValueError("offset exceeds report length")
    page = []
    for key, item in items[offset:offset + limit]:
        pointer = selector + "/" + str(key).replace("~", "~0").replace("/", "~1")
        preview = bounded_value(_report_preview(item))
        entry = {"key": key, "selector": pointer, "value": preview}
        if page and json_size(page + [entry]) > PAGE_CHARS:
            break
        page.append(entry)
    next_offset = offset + len(page)
    return {**base, "items": page, "total_items": len(items),
            "next_offset": next_offset if next_offset < len(items) else None,
            "note": "Partial fields have selectors for deeper reading. A page is not the whole report."}, []


def _crop_evidence(experiment_id, directory, target, region, output_root):
    if (not isinstance(region, list) or len(region) != 4
            or any(not isinstance(v, int) or isinstance(v, bool) for v in region)):
        raise ValueError("region must be integer [left, top, right, bottom] in source pixels")
    left, top, right, bottom = region
    with Image.open(target) as source:
        width, height = source.size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("region is outside source coordinates")
    if right - left > 1024 or bottom - top > 1024:
        raise ValueError("native-resolution region must be at most 1024x1024; request adjacent regions")
    root = Path(output_root) / "inspection"
    root.mkdir(parents=True, exist_ok=True)
    digest = sha256(json.dumps([experiment_id, region]).encode()).hexdigest()[:20]
    images, labels = [], []
    sources = [("original", Path(target)), ("overlay", directory / "result_annotation.png"),
               ("final_mask_after_user_constraints", directory / "mask.png")]
    for label, path in sources:
        if not path.exists():
            continue
        with Image.open(path) as source:
            if source.size != (width, height):
                raise ValueError("evidence image is not aligned with source coordinates")
            destination = root / f"{digest}_{label}.png"
            source.crop(region).save(destination)
        images.append(str(destination))
        labels.append(label)
    return {"experiment_id": experiment_id, "region_xyxy": region, "source_size": [width, height],
            "scale": 1, "image_order": labels,
            "note": "Aligned native-resolution crops; coordinates are in the original image. Check full images for global omissions."}, images
