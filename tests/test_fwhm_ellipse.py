import numpy as np

from analysis import calculate_radius, fit_fwhm_ellipse, fit_gaussian_center


def test_rotated_gaussian_fwhm_ellipse_and_overlap():
    y, x = np.indices((180, 220))
    angle = 0.48
    major = (x - 82.4) * np.cos(angle) + (y - 70.7) * np.sin(angle)
    minor = -(x - 82.4) * np.sin(angle) + (y - 70.7) * np.cos(angle)
    data = 7 + 4000 * np.exp(-0.5 * ((major / 21) ** 2 + (minor / 13) ** 2))
    _, _, initial, initial_angle, _, initial_major, initial_minor = calculate_radius(data)
    center = fit_gaussian_center(data, initial, initial_major,
                                 initial_minor, initial_angle)
    fitted = fit_fwhm_ellipse(data, center, initial_major)
    half_sigma_factor = np.sqrt(2 * np.log(2))
    np.testing.assert_allclose(
        [fitted['major'], fitted['minor']],
        [21 * half_sigma_factor, 13 * half_sigma_factor], atol=0.15)
    assert abs(fitted['angle'] - angle) < 0.02
    assert fitted['overlap'] > 0.97
    expected_x = 2 * half_sigma_factor / np.sqrt(
        (np.cos(angle) / 21) ** 2 + (np.sin(angle) / 13) ** 2)
    expected_y = 2 * half_sigma_factor / np.sqrt(
        (np.sin(angle) / 21) ** 2 + (np.cos(angle) / 13) ** 2)
    np.testing.assert_allclose([fitted['fwhm_x'], fitted['fwhm_y']],
                               [expected_x, expected_y], atol=0.2)

def test_overlap_decreases_for_nonelliptical_half_height_region():
    y, x = np.indices((180, 220))
    base = np.exp(-0.5 * (((x - 100) / 22) ** 2 + ((y - 90) / 14) ** 2))
    shoulder = 0.75 * np.exp(-0.5 * (((x - 121) / 12) ** 2 +
                                      ((y - 94) / 10) ** 2))
    data = 5 + 3000 * (base + shoulder)
    _, _, initial, angle, _, major, minor = calculate_radius(data)
    center = fit_gaussian_center(data, initial, major, minor, angle)
    fitted = fit_fwhm_ellipse(data, center, major)
    assert 0 <= fitted['overlap'] < 0.99
