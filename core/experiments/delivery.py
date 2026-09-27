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
    from core.experiments.drafts import atomic_json
    directory = Path(directory)
    files = ('mask.png', 'outputs.json', 'contours.json', 'measurements.json', 'result_annotation.png',
             'result_annotated.png', 'pipeline.json', 'quality_report.json', 'operator_trace.json')
    records = {name: {'path': name, 'sha256': sha256((directory / name).read_bytes()).hexdigest()}
               for name in files if (directory / name).is_file()}
    revision = sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    payload = {'schema_version': 1, **input_identity(image_path), 'revision': revision,
               'stage': 'final', 'rendering': rendering, 'files': records,
               'raw_outputs': 'raw_outputs.json', 'invalidated_outputs': list(invalidated),
               'stages': {'raw': {'outputs': 'raw_outputs.json'},
                          'constrained': {'mask': 'mask.png' if 'mask.png' in records else None},
                          'final': {'revision': revision, 'files': list(records)}}}
    atomic_json(directory / 'delivery_manifest.json', payload)
    return payload


def verify_manifest(directory, input_sha256):
    """Verify all committed delivery files before recovering an execution."""
    directory = Path(directory)
    payload = json.loads((directory / 'delivery_manifest.json').read_text(encoding='utf-8'))
    files = payload['files']
    required = {'outputs.json', 'contours.json', 'measurements.json', 'pipeline.json', 'quality_report.json'}
    if (payload.get('input_sha256') != input_sha256 or not required.issubset(files)
            or not {'result_annotation.png', 'result_annotated.png'}.intersection(files)
            or payload.get('revision') != sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()):
        raise ValueError('Delivery manifest identity or required artifacts do not match')
    for name, record in files.items():
        if name != Path(name).name or record['path'] != name:
            raise ValueError('Delivery artifact path is outside this experiment')
        if sha256((directory / name).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Delivery artifact integrity check failed')
    return payload
