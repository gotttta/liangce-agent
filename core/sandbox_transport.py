"""Bounded binary ndarray frames. No pickle, compression, or executable objects."""
import json
import math
import struct
import numpy as np

MAGIC = b'LCARRAY1'
DTYPES = {'|b1', '|u1', '<u2', '>u2', '<i4', '<i8', '<f4', '<f8'}


def encode_frame(value, max_bytes):
    buffers = []
    size = 0
    def encode(item):
        nonlocal size
        if isinstance(item, np.ndarray):
            data = np.ascontiguousarray(item)
            if data.dtype.str not in DTYPES:
                raise ValueError('unsupported transport dtype')
            size += data.nbytes
            if size > max_bytes:
                raise ValueError('array transport limit exceeded')
            index = len(buffers)
            buffers.append(data.tobytes())
            return {'__ndarray__': index, 'dtype': data.dtype.str, 'shape': list(data.shape), 'bytes': data.nbytes}
        if isinstance(item, dict):
            return {key: encode(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(val) for val in item]
        return item.item() if isinstance(item, np.generic) else item
    header = json.dumps(encode(value), allow_nan=False, separators=(',', ':')).encode()
    if len(header) + size + 12 > max_bytes:
        raise ValueError('transport limit exceeded')
    return MAGIC + struct.pack('!I', len(header)) + header + b''.join(buffers)


def decode_frame(raw, max_bytes):
    if len(raw) > max_bytes:
        raise ValueError('transport limit exceeded')
    if not raw.startswith(MAGIC):
        # Old trusted worker images return JSON; the new worker always emits v1.
        return json.loads(raw)
    if len(raw) < 12:
        raise ValueError('incomplete frame')
    length = struct.unpack('!I', raw[8:12])[0]
    if length > len(raw) - 12:
        raise ValueError('invalid header length')
    value = json.loads(raw[12:12 + length])
    offset, next_index = 12 + length, 0
    def decode(item):
        nonlocal offset, next_index
        if isinstance(item, dict) and '__ndarray__' in item:
            shape, dtype = item.get('shape'), item.get('dtype')
            if (item['__ndarray__'] != next_index or dtype not in DTYPES or not isinstance(shape, list)
                    or len(shape) > 4 or any(type(v) is not int or v < 0 for v in shape)):
                raise ValueError('invalid array descriptor')
            size = math.prod(shape) * np.dtype(dtype).itemsize
            if item.get('bytes') != size or size > len(raw) - offset:
                raise ValueError('invalid array size')
            array = np.frombuffer(raw, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape).copy()
            offset += size
            next_index += 1
            return array
        if isinstance(item, dict):
            return {key: decode(val) for key, val in item.items()}
        if isinstance(item, list):
            return [decode(val) for val in item]
        return item
    result = decode(value)
    if offset != len(raw):
        raise ValueError('trailing frame bytes')
    return result
