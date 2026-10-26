import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize

from analysis import calculate_radius, fit_fwhm_ellipse, fit_gaussian_center
from app_gui import create_ellipse_figure, create_ellipse_zoom_figure


def rotated_spot(shape=(180, 220), center=(82.4, 70.7), angle=0.48):
    y, x = np.indices(shape)
    dx, dy = x - center[0], y - center[1]
    major = dx * np.cos(angle) + dy * np.sin(angle)
    minor = -dx * np.sin(angle) + dy * np.cos(angle)
    return 7 + 4000 * np.exp(-0.5 * ((major / 21) ** 2 + (minor / 13) ** 2))


def test_gaussian_fit_finds_subpixel_center_and_rotation():
    data = rotated_spot()
    _, _, initial, angle, _, major, minor = calculate_radius(data)
    fitted = fit_gaussian_center(data, initial, major, minor, angle)
    np.testing.assert_allclose(fitted, (82.4, 70.7), atol=0.1)
    assert abs(angle - 0.48) < 0.03


def test_both_spot_views_use_fitted_center_as_micron_origin():
    data = rotated_spot()
    _, _, initial, angle, _, major, minor = calculate_radius(data)
    center = fit_gaussian_center(data, initial, major, minor, angle)
    for make_figure in (create_ellipse_figure, create_ellipse_zoom_figure):
        figure = make_figure(major, minor, center, angle, data, 3, 0.44)
        axes = figure.axes[0]
        extent = axes.images[0].get_extent()
        np.testing.assert_allclose(extent[0] + (center[0] + 0.5) * 0.44, 0)
        np.testing.assert_allclose(extent[3] + (center[1] + 0.5) * 0.44, 0)
        np.testing.assert_allclose([np.mean(axes.get_xlim()),
                                    np.mean(axes.get_ylim())], (0, 0), atol=1e-12)
        np.testing.assert_allclose([axes.lines[0].get_xdata()[0],
                                    axes.lines[0].get_ydata()[0]], (0, 0))
        assert type(axes.images[0].norm) is Normalize
        assert 'μm' in axes.get_xlabel() and 'μm' in axes.get_ylabel()
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        colorbar_axes = figure.axes[1]
        assert abs(axes.get_window_extent(renderer).height
                   - colorbar_axes.get_window_extent(renderer).height) < 1
        assert colorbar_axes.yaxis.label.get_window_extent(renderer).x1 < figure.bbox.x1


def test_spot_views_have_distinct_overlay_roles():
    data = rotated_spot()
    _, _, initial, angle, _, major, minor = calculate_radius(data)
    center = fit_gaussian_center(data, initial, major, minor, angle)
    fitted = fit_fwhm_ellipse(data, center, major)
    scale = np.sqrt(2 / np.log(2))
    major_e2 = fitted['major'] * scale
    minor_e2 = fitted['minor'] * scale

    full = create_ellipse_figure(major_e2, minor_e2, center, fitted['angle'],
                                 data, 3, 0.44, fitted)
    full_labels = {line.get_label() for line in full.axes[0].lines}
    assert 'FWHM ellipse fit' in full_labels
    assert 'Integration region' in full_labels
    assert r'$1/e^2$ from FWHM' not in full_labels
    full_colors = {line.get_label(): line.get_color()
                   for line in full.axes[0].lines}
    assert full_colors['FWHM ellipse fit'] == 'white'
    assert full_colors['Integration region'] == '#ff7f0e'

    detail = create_ellipse_zoom_figure(
        major_e2, minor_e2, center, fitted['angle'], data, 3, 0.44, fitted)
    detail_labels = {text.get_text() for text in detail.axes[0].get_legend().texts}
    assert detail_labels == {r'$1/e^2$ contour', 'FWHM contour'}
    detail_colors = {line.get_color()
                     for line in detail.axes[0].get_legend().get_lines()}
    assert detail_colors == {'white', '#ff7f0e'}
