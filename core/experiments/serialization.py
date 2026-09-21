"""Convert experiment data to JSON values without stringifying arrays or numbers."""
from pathlib import Path

import numpy as np


def to_jsonable(value):
    """Return a serializable copy while leaving live execution artifacts intact."""
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    # Unknown objects must still fail JSON encoding, not silently become strings.
    return value
