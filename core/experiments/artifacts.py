"""Bounded intermediate artifact transport and durable experiment manifests."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.operators import ImageArtifact, MaskArtifact, MetadataArtifact, ContourArtifact
from core.experiments.serialization import to_jsonable

MAX_ARTIFACTS = 8
MAX_BYTES = 32 * 1024 * 1024


def export_artifacts(artifacts, final=False):
    result = {}
    used = 0
    for name, value in artifacts.items():
        if name in {'image', '$image'} and not final:
            continue
        if isinstance(value, MetadataArtifact):
            item = {'kind': 'metadata', 'data': value.data}
            size = len(json.dumps(value.data, allow_nan=False).encode())
        elif isinstance(value, ContourArtifact):
            item = {'kind': 'contours', 'data': [x.tolist() for x in value.contours],
                    'shape': list(value.image_shape), 'metadata': value.metadata}
            size = sum(x.nbytes for x in value.contours)
        elif isinstance(value, (ImageArtifact, MaskArtifact)):
            item = {'kind': 'mask' if isinstance(value, MaskArtifact) else 'image',
                    'data': value.data, 'metadata': value.metadata}
            size = value.data.nbytes
        else:
            continue
        if not final and (len(result) >= MAX_ARTIFACTS or used + size > MAX_BYTES):
            continue
        result[name] = item
        used += size
    return result


def import_artifacts(payload):
    if not isinstance(payload, dict):
        raise ValueError('artifacts must be an object')
    result = {}
    for name, item in payload.items():
        if not isinstance(item, dict) or not isinstance(item.get('metadata', {}), dict):
            raise ValueError('invalid artifact')
        kind = item.get('kind')
        if kind == 'metadata':
            if not isinstance(item['data'], dict):
                raise ValueError('metadata output must be an object')
            json.dumps(item['data'], allow_nan=False)
            result[name] = MetadataArtifact(item['data'])
        elif kind == 'contours':
            contours = tuple(np.asarray(x, dtype=np.float32) for x in item['data'])
            if any(not np.isfinite(x).all() for x in contours):
                raise ValueError('invalid contours')
            result[name] = ContourArtifact(contours, tuple(item['shape']), item.get('metadata', {}))
        elif kind in {'mask', 'image'}:
            data = np.asarray(item['data'], dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError('non-finite artifact')
            if kind == 'mask' and not np.isin(data, [0, 1]).all():
                raise ValueError('non-binary mask')
            result[name] = (MaskArtifact if kind == 'mask' else ImageArtifact)(data, item.get('metadata', {}))
        else:
            raise ValueError('unknown artifact kind')
    return result


def persist_artifacts(execution, directory):
    directory = Path(directory).resolve()
    output = directory / 'artifacts'
    output.mkdir(exist_ok=True)
    records = []
    for index, (node, item) in enumerate(export_artifacts(execution.artifacts).items()):
        data = item['data']
        if item['kind'] not in {'image', 'mask'}:
            continue
        stem = f'node_{index:02d}'  # Never use model-supplied node IDs as filenames.
        raw = output / f'{stem}.npy'
        preview = output / f'{stem}.png'
        np.save(raw, data, allow_pickle=False)
        finite = np.isfinite(data)
        low = float(data[finite].min()) if finite.any() else 0.0
        high = float(data[finite].max()) if finite.any() else 0.0
        if item['kind'] == 'mask':
            pixels = data.astype(np.uint8) * 255
        else:
            scaled = (np.where(finite, data, low) - low) / (high - low) if high > low else np.zeros_like(data)
            pixels = np.clip(scaled * 255, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels)
        image.thumbnail((1024, 1024))
        image.save(preview)
        records.append({
            'id': sha256(f'{directory}:{node}'.encode()).hexdigest()[:24],
            'node_id': node, 'kind': item['kind'], 'shape': list(data.shape),
            'stage': 'pipeline_output_before_user_constraints',
            'raw_path': str(raw), 'preview_path': str(preview),
            'sha256': sha256(raw.read_bytes()).hexdigest(),
            'preview_scale_xy': [image.width / data.shape[1], image.height / data.shape[0]],
            'display_range': [low, high],
        })
    return records


def experiment_scope(image_path, contract=None, context=None):
    """Only compare/reuse acceptance under identical inputs and requirements."""
    from core.task_contract import FIELDS
    context = context or {}
    contract = contract or context.get('task_contract') or {}
    def digest(path):
        return sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
    masks = {}
    for key in ('ground_truth_mask_path', 'include_mask_path', 'exclude_mask_path',
                'false_positive_mask_path', 'false_negative_mask_path'):
        path = context.get(key) or (context.get('human_feedback') or {}).get(key) or (context.get('feedback') or {}).get(key)
        masks[key] = digest(path)
    return {'input_sha256': digest(image_path), 'coordinate_version': 'stored-pixels-v1',
            'contract': {key: contract.get(key) for key in (*FIELDS, 'unit')}, 'masks': masks}


def update_experiment(attempt, **changes):
    """Persist lifecycle updates without overwriting immutable execution evidence."""
    directory = attempt.get('directory')
    path = Path(directory) / 'experiment.json' if directory else None
    # Older checkpoints and externally supplied attempts may predate manifests.
    if path is None or not path.is_file():
        attempt.update(to_jsonable(changes))
        return
    record = json.loads(path.read_text(encoding='utf-8'))
    record.update(to_jsonable(changes))
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    attempt.update(to_jsonable(changes))


def record_experiments(attempts, image_path, previous_state, iteration, *, scope=None):
    """Write every attempt, including invalid and failed candidates, atomically."""
    image_hash = sha256(Path(image_path).read_bytes()).hexdigest()
    for attempt in attempts:
        directory = Path(attempt['directory']).resolve()
        experiment_id = sha256(str(directory).encode()).hexdigest()[:24]
        attempt['experiment_id'] = experiment_id
        attempt.update(
            algorithm_version=sha256(json.dumps(to_jsonable(attempt.get('pipeline') or {}), sort_keys=True).encode()).hexdigest(),
            parent_experiment_id=(previous_state or {}).get('selected_experiment_id'),
            change_reason=attempt.get('change_reason') or attempt.get('hypothesis', ''),
            expected_change=attempt.get('expected_change', ''),
            acceptance_status='pending',
            scope=scope or experiment_scope(image_path, context=previous_state),
        )
        record = {key: value for key, value in attempt.items() if key != 'execution'}
        if attempt.get('execution') is not None:
            record['operator_trace'] = list(attempt['execution'].trace)
        record['feedback_constraints'] = {key: (previous_state or {}).get(key) for key in (
            'human_feedback', 'include_mask_path', 'exclude_mask_path',
            'false_positive_mask_path', 'false_negative_mask_path')}
        record.update(schema_version=1, input_sha256=image_hash, coordinate_version='stored-pixels-v1', iteration=iteration,
                      parent_experiment_id=(previous_state or {}).get('selected_experiment_id'),
                      acceptance_status='pending',
                      execution_status='failed' if attempt['status'] in {'failed', 'duplicate_pipeline'} else 'completed')
        path = directory / 'experiment.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(to_jsonable(record), ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
