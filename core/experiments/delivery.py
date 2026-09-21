"""Build one version of final artifacts after pixel constraints are applied."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from core.operators import MaskArtifact, ContourArtifact, build_default_registry
from core.experiments.artifacts import export_artifacts
from core.experiments.serialization import to_jsonable
from core.input_contract import input_identity


def finalize_mask_outputs(execution, mask):
    original = execution.mask
    changed = not np.array_equal(original.data, mask)
    execution.mask = MaskArtifact(mask.copy(), {**original.metadata, 'stage': 'final'})
    execution.contours = build_default_registry().run('extract_contours', execution.mask).artifact
    outputs, invalidated = {}, []
    for name, artifact in execution.outputs.items():
        if artifact is original:
            outputs[name] = execution.mask
        elif isinstance(artifact, ContourArtifact):
            outputs[name] = execution.contours
        elif changed:
            # Without a declared recomputation graph these outputs may depend on
            # the old mask. Never present them as corrected measurements/images.
            invalidated.append(name)
        else:
            outputs[name] = artifact
    outputs.setdefault('mask', execution.mask)
    outputs.setdefault('contours', execution.contours)
    execution.outputs = outputs
    return invalidated


def write_outputs(execution, directory, filename='outputs.json'):
    directory = Path(directory)
    payload = to_jsonable(export_artifacts(execution.outputs, final=True))
    path = directory / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    # State carries artifact references, never pixel/contour arrays.
    return {name: {'kind': item['kind'], 'path': str(path), 'selector': name,
                   'stage': 'final' if filename == 'outputs.json' else 'raw',
                   **({'fields': list(item['data'])} if item['kind'] == 'metadata' else {}),
                   'shape': list(execution.outputs[name].data.shape) if item['kind'] in {'image', 'mask'} else item.get('shape'),
                   **({'data': item['data']} if item['kind'] == 'metadata' and len(json.dumps(item['data'])) < 4096 else {})}
            for name, item in payload.items()}


def write_manifest(directory, image_path, rendering, invalidated=()):
    directory = Path(directory)
    files = ('mask.png', 'outputs.json', 'contours.json', 'measurements.json', 'result_annotation.png', 'result_annotated.png')
    records = {name: {'path': name, 'sha256': sha256((directory / name).read_bytes()).hexdigest()}
               for name in files if (directory / name).is_file()}
    revision = sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    payload = {'schema_version': 1, **input_identity(image_path), 'revision': revision,
               'stage': 'final', 'rendering': rendering, 'files': records,
               'raw_outputs': 'raw_outputs.json', 'invalidated_outputs': list(invalidated),
               'stages': {'raw': {'outputs': 'raw_outputs.json'},
                          'constrained': {'mask': 'mask.png' if 'mask.png' in records else None},
                          'final': {'revision': revision, 'files': list(records)}}}
    (directory / 'delivery_manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload
