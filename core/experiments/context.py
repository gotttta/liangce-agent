"""Model-facing evidence views. Durable reports remain the source of truth."""
import json
from pathlib import Path

import numpy as np


METADATA_CHARS = 4000
OUTPUTS_CHARS = 16000


def json_size(value):
    return len(json.dumps(value, ensure_ascii=False))


def bounded_value(value, *, limit=METADATA_CHARS):
    """Keep small values exact; explicitly mark partial tables and large fields.

    This is only for optional artifact data, never user requirements, errors,
    acceptance criteria, or executable operator source.
    """
    if json_size(value) <= limit:
        return value
    if isinstance(value, dict):
        kept, omitted = {}, []
        # Small scalar facts (including units) precede large arrays/tables.
        for key in sorted(value, key=lambda key: json_size(value[key])):
            preview = bounded_value(value[key], limit=min(1200, limit // 2))
            if json_size({**kept, key: preview}) <= limit - 300:
                kept[key] = preview
            else:
                omitted.append(key)
        return {"partial": True, "fields": kept, "omitted_field_count": len(omitted),
                "omitted_fields": omitted[:20], "note": "Read the report selector for full details; omitted is not absent."}
    if isinstance(value, list):
        numeric = {}
        if value and all(isinstance(item, dict) for item in value):
            keys = sorted({key for item in value for key, val in item.items()
                           if isinstance(val, (int, float)) and not isinstance(val, bool)})
            for key in keys[:6]:
                entries = [(i, float(item[key])) for i, item in enumerate(value)
                           if isinstance(item.get(key), (int, float)) and np.isfinite(item[key])]
                if entries:
                    indices, vals = zip(*entries)
                    numeric[key] = {'min': min(vals), 'max': max(vals), 'mean': float(np.mean(vals)),
                        'min_index': indices[int(np.argmin(vals))], 'max_index': indices[int(np.argmax(vals))]}
        anomalies = [i for i, item in enumerate(value) if isinstance(item, dict) and
                     (item.get('status') in {'failed', 'invalid', 'error'} or item.get('issues') or item.get('error'))]
        indices = list(dict.fromkeys([*anomalies[:3], *[v['max_index'] for v in numeric.values()], 0, len(value) - 1]))
        result = {'partial': True, 'count': len(value), 'statistics': numeric,
                  'anomaly_indices': anomalies[:10], 'sample_indices': [], 'sample': [],
                  'note': 'Samples include anomalies/extremes; read exact selectors for full details.'}
        for index in indices:
            if index >= 0 and json_size(result) + json_size(value[index]) < limit - 50:
                result['sample_indices'].append(index)
                result['sample'].append(value[index])
        if json_size(result) > limit:
            result['statistics'] = {}
        return result
    return {"partial": True, "type": type(value).__name__, "chars": json_size(value),
            "note": "Value retained in the report; request its selector."}


def artifact_view(artifact):
    kind = artifact.get("kind")
    if 'data' not in artifact and artifact.get('path'):
        return dict(artifact)
    result = {"kind": kind}
    data = artifact.get("data")
    if kind in {"mask", "image"}:
        array = np.asarray(data)
        result.update(shape=list(array.shape), dtype=str(array.dtype), elements=int(array.size))
        if array.size and (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_):
            finite = np.isfinite(array)
            result["nonfinite_count"] = int(array.size - np.count_nonzero(finite))
            if finite.any():
                result.update(min=float(array[finite].min()), max=float(array[finite].max()))
            if kind == "mask":
                result["foreground_pixels"] = int(np.count_nonzero(array))
        result["data_omitted"] = "Pixel array stored in outputs.json; inspect images or a native-resolution ROI."
    elif kind == "contours":
        result.update(shape=artifact.get("shape"), count=len(data or []),
                      point_count=sum(len(contour) for contour in (data or [])),
                      data_omitted="Coordinates stored in outputs.json; inspect the overlay/ROI.")
    else:
        result["data"] = bounded_value(data)
    if artifact.get("metadata"):
        result["metadata"] = bounded_value(artifact["metadata"], limit=1200)
    return result


def quality_for_model(quality, directory=None):
    """Keep quality/error/constraint facts exact, summarize only artifact payloads."""
    result = {key: value for key, value in (quality or {}).items() if key != "outputs"}
    outputs, omitted = {}, []
    for name, artifact in (quality or {}).get("outputs", {}).items():
        view = artifact_view(artifact)
        # JSON Pointer identifies exact metadata even when output names contain '/'.
        view["selector"] = "/" + name.replace("~", "~0").replace("/", "~1")
        if json_size({**outputs, name: view}) <= OUTPUTS_CHARS:
            outputs[name] = view
        else:
            omitted.append(name)
    if outputs:
        result["outputs"] = outputs
    if omitted:
        result["outputs_omitted"] = {"count": len(omitted), "names": omitted[:20]}
    if (quality or {}).get("outputs"):
        result["output_evidence"] = {
            "report": "outputs", "path": str(Path(directory) / "outputs.json") if directory else None,
            "stage": "final",
            "note": "Final outputs include user constraints. Raw results are retained separately in raw_outputs.json.",
        }
    return result


def candidate_for_model(attempt):
    return {
        "experiment_id": attempt.get("experiment_id"), "name": attempt.get("name"),
        "status": attempt.get("status"), "hypothesis": attempt.get("hypothesis", ""),
        **{key: attempt[key] for key in ('change_reason', 'expected_change', 'parent_experiment_id',
                                         'acceptance_status', 'review') if key in attempt},
        "facts": quality_for_model(attempt.get("quality", {}), attempt.get("directory")),
        "measurement_summary": (attempt.get("measurements") or {}).get("summary", {}),
        "evidence": {"reports": ["measurements", "outputs", "pipeline", "operator_trace", "quality_report"],
                     "inspection_tool": "inspect_experiment",
                     "note": "Use experiment_id for exact report pages or aligned original/overlay/final-mask crops."},
    }
