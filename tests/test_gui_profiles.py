import numpy as np

from app_gui import (calculate_a0_axis_cuts, calculate_a0_cuts,
                     create_a0_axis_profile_figure, create_a0_profile_figure)


def gaussian_spot():
    y, x = np.indices((160, 200))
    return 10 + 4000 * np.exp(-0.5 * (((x - 100) / 20) ** 2
                                     + ((y - 80) / 12) ** 2))


def test_a0_cuts_cross_center_and_use_micron_positions():
    data = gaussian_spot()
    cuts = calculate_a0_cuts(data, (100, 80), 0.44, 2.0)
    x_positions, x_a0, y_positions, y_a0 = cuts
    reference = np.mean(data[79:82, 99:102])
    np.testing.assert_allclose(x_positions[100], 0)
    np.testing.assert_allclose(y_positions[80], 0)
    np.testing.assert_allclose(x_a0[100], 2 * np.sqrt(data[80, 100] / reference))
    np.testing.assert_allclose(y_a0[80], x_a0[100])
    assert x_a0[100] > x_a0[60]
    figure = create_a0_profile_figure(cuts, 40, 24, 0.44)
    assert 'μm' in figure.axes[0].get_xlabel()
    assert figure.axes[0].get_ylabel() == '$a_0$'


def test_a0_axis_cuts_follow_rotated_major_and_minor_axes():
    y, x = np.indices((160, 200))
    angle = 0.55
    dx, dy = x - 100, y - 80
    major = dx * np.cos(angle) + dy * np.sin(angle)
    minor = -dx * np.sin(angle) + dy * np.cos(angle)
    data = 10 + 4000 * np.exp(-0.5 * ((major / 20) ** 2
                                      + (minor / 10) ** 2))
    cuts = calculate_a0_axis_cuts(data, (100, 80), angle, 0.44, 2.0)
    major_positions, major_a0, minor_positions, minor_a0 = cuts
    major_center = np.flatnonzero(major_positions == 0)[0]
    minor_center = np.flatnonzero(minor_positions == 0)[0]
    np.testing.assert_allclose(major_a0[major_center], minor_a0[minor_center])
    assert np.interp(4.4, major_positions, major_a0) > np.interp(
        4.4, minor_positions, minor_a0)

    figure = create_a0_axis_profile_figure(cuts, 40, 20, 0.44)
    labels = {line.get_label() for line in figure.axes[0].lines}
    assert 'major-axis cut' in labels
    assert 'minor-axis cut' in labels
    assert 'μm' in figure.axes[0].get_xlabel()
    assert figure.axes[0].get_ylabel() == '$a_0$'
