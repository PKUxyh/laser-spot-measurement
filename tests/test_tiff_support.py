import numpy as np
import pandas as pd
import pytest
import tifffile

from analysis import calculate_radius, preprocess_data
from data_io import load_spot_data
from zernike_analysis import zernike_analysis


def test_csv_and_tiff_give_the_same_spot_measurements(tmp_path):
    y, x = np.indices((80, 100))
    counts = np.rint(4000 * np.exp(-2 * (((x - 51) / 14) ** 2 + ((y - 39) / 9) ** 2))).astype(np.uint16)
    csv_path = tmp_path / 'spot.csv'
    tiff_path = tmp_path / 'spot.tiff'
    pd.DataFrame(counts).to_csv(csv_path, index=False)
    tifffile.imwrite(tiff_path, counts)

    csv_data = load_spot_data(csv_path)
    tiff_data = load_spot_data(tiff_path)
    np.testing.assert_array_equal(csv_data, tiff_data)
    assert tiff_data.max() == counts.max()
    np.testing.assert_allclose(calculate_radius(preprocess_data(csv_data))[:2],
                               calculate_radius(preprocess_data(tiff_data))[:2])


def test_color_tiff_is_rejected(tmp_path):
    path = tmp_path / 'color.tif'
    tifffile.imwrite(path, np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='single grayscale 2D image'):
        load_spot_data(path)


def test_zernike_reconstruction_keeps_image_shape(tmp_path):
    y, x = np.indices((40, 40))
    counts = np.rint(1000 * np.exp(-2 * (((x - 20) / 7) ** 2 + ((y - 20) / 7) ** 2))).astype(np.uint16)
    path = tmp_path / 'spot.tif'
    tifffile.imwrite(path, counts)
    data = load_spot_data(path)
    coeffs, zernike = zernike_analysis(data)
    reconstruction = zernike.eval_grid(coeffs, matrix=True)
    assert reconstruction.shape == data.shape
    assert np.isfinite(reconstruction).all()
