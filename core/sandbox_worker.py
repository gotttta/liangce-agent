"""Container-only JSON entrypoint. Never run this module on the host."""
import json
import os
import sys
import numpy as np
from core.sandbox import MAX_INPUT_BYTES, MAX_OUTPUT_BYTES, require_docker_worker, _serialize_result
from core.pipelines.dsl import execute_pipeline
from core.sandbox_transport import encode_frame, decode_frame


def main():
    require_docker_worker()
    # Preserve the protocol channel, redirect algorithm print/native stdout to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'wb')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError('input limit exceeded')
        request = decode_frame(raw, MAX_INPUT_BYTES)
        result = execute_pipeline(np.asarray(request['image'], dtype=np.float32),
                                  request['pipeline'], allow_generated=True, inputs=request.get('inputs'))
        payload = {'ok': True, 'result': _serialize_result(result)}
        protocol.write(encode_frame(payload, MAX_OUTPUT_BYTES))
    except BaseException as exc:
        protocol.write(encode_frame({'ok': False, 'code': 'resource_limit' if isinstance(exc, MemoryError) else 'execution_failed',
                   'error': f'{type(exc).__name__}: {exc}'[:2000]}, MAX_OUTPUT_BYTES))
    finally:
        protocol.flush()


if __name__ == '__main__':
    main()
