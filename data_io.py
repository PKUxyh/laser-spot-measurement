from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


SUPPORTED_SUFFIXES = {'.csv', '.tif', '.tiff'}


def load_spot_data(file_path):
    """Load a single two-dimensional intensity image without rescaling counts."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        data = pd.read_csv(path).to_numpy(dtype=np.float64)
    elif suffix in {'.tif', '.tiff'}:
        data = tifffile.imread(path)
    else:
        raise ValueError(f'Unsupported file format: {path.suffix}')

    if data.ndim != 2:
        raise ValueError(f'Expected a single grayscale 2D image, got shape {data.shape}')
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f'Expected numeric intensity data, got {data.dtype}')
    return np.asarray(data, dtype=np.float64)
