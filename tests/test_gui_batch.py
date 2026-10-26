from pathlib import Path

import numpy as np
import pytest
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg

import app_gui
from app_gui import (analyze_spot_file, create_average_spot_figure,
                     create_batch_a0_figure,
                     create_batch_waists_figure,
                     create_pointing_jitter_figure,
                     natural_path_key, padded_limits,
                     summarize_batch_results)


def spot_result(name, a0, x_radius, y_radius, center, value):
    return {
        'file_path': Path(name),
        'data': np.full((8, 10), value, dtype=float),
        'radii': (x_radius, y_radius, center, 0.0, 1.0, x_radius, y_radius),
        'a0': a0,
    }


def test_shot_files_use_natural_numeric_order():
    names = [Path('shot_10.tif'), Path('shot_2.tif'), Path('shot_1.tif')]
    assert [path.name for path in sorted(names, key=natural_path_key)] == [
        'shot_1.tif', 'shot_2.tif', 'shot_10.tif']


def test_folder_entry_analyzes_only_the_user_selected_subset(tmp_path, monkeypatch):
    for name in ('shot_10.tif', 'shot_2.tif', 'shot_1.csv', 'notes.txt'):
        (tmp_path / name).write_bytes(b'placeholder')

    class DummyApp:
        busy = False
        root = None

        def _choose_folder_files(self, folder_path, file_paths):
            assert folder_path == tmp_path
            assert [path.name for path in file_paths] == [
                'shot_1.csv', 'shot_2.tif', 'shot_10.tif']
            return file_paths[1:]

        def analyze_batch(self, file_paths):
            self.analyzed = file_paths

    dummy = DummyApp()
    monkeypatch.setattr(app_gui.filedialog, 'askdirectory',
                        lambda **kwargs: str(tmp_path))
    app_gui.FocusAnalysisApp.select_folder(dummy)
    assert [path.name for path in dummy.analyzed] == ['shot_2.tif', 'shot_10.tif']


def test_batch_summary_uses_sample_standard_deviation_and_micron_centers():
    results = [
        spot_result('shot_1.tif', 2.0, 10, 6, (20, 30), 1),
        spot_result('shot_2.tif', 4.0, 14, 8, (24, 26), 3),
    ]
    results[0]['shot_number'] = 1
    results[1]['shot_number'] = 3
    requested = [Path('shot_1.tif'), Path('shot_2.tif')]
    batch = summarize_batch_results(results, requested, [], 0.5)

    np.testing.assert_allclose(batch['a0_values'], [2, 4])
    np.testing.assert_array_equal(batch['shot_numbers'], [1, 3])
    np.testing.assert_allclose(batch['wx_values'], [5, 7])
    np.testing.assert_allclose(batch['wy_values'], [3, 4])
    np.testing.assert_allclose(batch['center_offsets_um'], [[-1, 1], [1, -1]])
    np.testing.assert_allclose(batch['mean_data'], 2)
    np.testing.assert_allclose(batch['mean_center_pixel'], (22, 28))
    assert batch['stats']['a0_mean'] == 3
    np.testing.assert_allclose(batch['stats']['a0_std'], np.sqrt(2))
    np.testing.assert_allclose(batch['stats']['jitter_x'], np.sqrt(2))
    np.testing.assert_allclose(batch['stats']['jitter_y'], np.sqrt(2))
    np.testing.assert_allclose(batch['stats']['jitter_rms'], np.sqrt(2))


def test_batch_summary_rejects_mismatched_image_sizes():
    results = [
        spot_result('shot_1.tif', 2.0, 10, 6, (20, 30), 1),
        spot_result('shot_2.tif', 2.0, 10, 6, (20, 30), 1),
    ]
    results[1]['data'] = np.ones((9, 10))
    with pytest.raises(ValueError, match='matching image sizes'):
        summarize_batch_results(results, [Path('shot_1.tif'),
                                          Path('shot_2.tif')], [], 0.5)


def test_batch_figures_render_the_requested_four_views():
    shots = np.arange(1, 4)
    figures = [
        create_batch_a0_figure(shots, np.array([1.8, 2.0, 2.1])),
        create_batch_waists_figure(shots, np.array([5.0, 5.2, 4.9]),
                                   np.array([3.0, 3.1, 2.9])),
        create_pointing_jitter_figure(np.array([[-0.2, 0.1],
                                                [0.0, -0.1],
                                                [0.2, 0.0]])),
        create_average_spot_figure(np.arange(80, dtype=float).reshape(8, 10),
                                   (4.5, 3.5), 0.5),
    ]
    for figure in figures:
        FigureCanvasAgg(figure).draw()

    assert figures[0].axes[0].get_xlabel() == 'Shot'
    assert figures[0].axes[0].get_ylabel() == '$a_0$'
    assert figures[1].axes[0].get_xlabel() == 'Shot'
    assert len(figures[1].axes[0].lines) == 2
    assert 'μm' in figures[2].axes[0].get_xlabel()
    assert len(figures[3].axes) == 2  # image and colorbar


def test_batch_series_y_limits_pad_the_current_data_range():
    shots = np.arange(1, 4)
    a0_values = np.array([4.8, 4.9, 5.0])
    a0_figure = create_batch_a0_figure(shots, a0_values)
    lower, upper = a0_figure.axes[0].get_ylim()
    assert 0 < lower < np.min(a0_values)
    assert upper > np.max(a0_values)

    waist_figure = create_batch_waists_figure(
        shots, np.array([5.0, 5.1, 5.2]), np.array([3.0, 3.1, 3.2]))
    lower, upper = waist_figure.axes[0].get_ylim()
    assert 0 < lower < 3.0
    assert upper > 5.2

    constant_lower, constant_upper = padded_limits([2.0, 2.0])
    assert constant_lower < 2.0 < constant_upper


def test_batch_pipeline_analyzes_multiple_tiffs(tmp_path):
    y, x = np.indices((100, 120))
    paths = []
    for shot, center_x in enumerate((58, 62), start=1):
        counts = np.rint(4000 * np.exp(-0.5 * (
            ((x - center_x) / 14) ** 2 + ((y - 50) / 9) ** 2))).astype(np.uint16)
        path = tmp_path / f'shot_{shot}.tif'
        tifffile.imwrite(path, counts)
        paths.append(path)
    params = {
        'PIXEL_WIDTH_MICRONS': 0.44,
        'region_total': 8,
        'energy': 2.66,
        'duration': 35,
        'lambda_L': 800,
    }
    results = [analyze_spot_file(path, params) for path in paths]
    batch = summarize_batch_results(results, paths, [], 0.44)

    assert len(batch['a0_values']) == 2
    assert batch['stats']['jitter_x'] > 0
    np.testing.assert_allclose(batch['stats']['jitter_y'], 0, atol=1e-6)
    assert batch['mean_data'].shape == (100, 120)
