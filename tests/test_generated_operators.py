import numpy as np
import pytest

from core.operators import build_default_registry, ImageArtifact
from core.operators.generated import validate_generated_source
from core.pipelines.dsl import execute_pipeline, validate_pipeline
from core.sandbox import execute_pipeline_sandbox

CUSTOM = {
    'name': 'bright_custom', 'input_artifact': 'ImageArtifact',
    'output_artifact': 'MaskArtifact',
    'source': 'def apply(data, params):\n    return data > np.mean(data)',
}


def pipeline_for(source=None):
    spec = dict(CUSTOM)
    if source is not None:
        spec['source'] = source
    return {'generated_operators': [spec], 'steps': [
        {'id': 'final_mask', 'op': 'bright_custom', 'input': 'image', 'params': {}}]}


def test_validation_accepts_imports_helpers_loops_and_fft_without_execution(tmp_path):
    sentinel = tmp_path / 'must_not_exist'
    source = f'''import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
open({str(sentinel)!r}, 'w').write('executed')
def helper(data):
    for i in range(2):
        data = gaussian_filter(data, 1)
    return np.fft.ifft2(np.fft.fft2(data)).real
def apply(data, params):
    return helper(data) > np.mean(data)
'''
    validate_pipeline(pipeline_for(source))
    assert not sentinel.exists()


@pytest.mark.parametrize('allow_generated', [False, True])
def test_host_execution_cannot_bypass_docker(allow_generated):
    with pytest.raises(ValueError, match='sandbox'):
        execute_pipeline(np.zeros((4, 4)), pipeline_for(), allow_generated=allow_generated)


def test_direct_registry_execution_also_requires_docker():
    registry = build_default_registry([CUSTOM])
    with pytest.raises(ValueError, match='Docker sandbox'):
        registry.run('bright_custom', ImageArtifact(np.zeros((4, 4))))


@pytest.mark.parametrize('source', ['', 'def apply(:', 'def other(a,b): return a',
                                    'def apply(data): return data'])
def test_interface_is_still_validated(source):
    with pytest.raises(ValueError):
        validate_generated_source(source)


def test_complete_algorithm_can_be_declared_non_atomic():
    pipeline = pipeline_for()
    pipeline['generated_operators'][0]['atomic'] = False
    validate_pipeline(pipeline)


def test_generated_operator_runs_in_docker(docker_sandbox):
    image = np.zeros((12, 12), dtype=np.float32)
    image[3:6, 4:7] = 10
    result = execute_pipeline_sandbox(image, pipeline_for('''import cv2
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
def helper(x):
    for _ in range(2):
        x = gaussian_filter(x, 0)
    return np.fft.ifft2(np.fft.fft2(x)).real
def apply(data, params):
    data = helper(data)
    return data > threshold_otsu(data)
'''))
    assert result.mask.data.sum() == 9
