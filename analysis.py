from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from scipy.ndimage import gaussian_filter, label as ndimage_label
from matplotlib.colors import LogNorm  # new import
from scipy.signal import convolve2d
from scipy.optimize import least_squares
from skimage.measure import find_contours
from matplotlib.path import Path as MplPath
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from data_io import load_spot_data

# 设置全局样式
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 1,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'figure.dpi': 300,
    'figure.figsize': (6.74, 6.74/1.6),
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.framealpha': 1,
    'legend.fontsize': 12,
})
# # #双色
rgb1 = '#3F77A3'
rgb2 = '#E49A5C'
rgb3 = '#EC3232'
rgb4 = '#5CE49A'


#############################SP620U 4.4um per pixel     *10倍
PIXEL_WIDTH_MICRONS = 4.4 / 39.11 # 单像素宽度为4.4微米
energy = 3.3*0.5585
duration = 35e-15
lambda_L = 800e-9
dx = PIXEL_WIDTH_MICRONS * 1e-4
dy = PIXEL_WIDTH_MICRONS * 1e-4
region_total = 8

def preprocess_data(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data[data < 0] = 0
    # data = gaussian_filter(data, sigma=7)      #高斯滤波不合适，会让计算偏差很大
    return data

def calculate_rotated_coordinates(x_center, y_center, x_indices, y_indices, orientation):
    x_rot = (x_indices - x_center) * np.cos(orientation) + (y_indices - y_center) * np.sin(orientation)
    y_rot = -(x_indices - x_center) * np.sin(orientation) + (y_indices - y_center) * np.cos(orientation)
    return x_rot, y_rot

def calculate_radius(data):
    data = preprocess_data(data)
    sum_filter = np.ones((5,5), dtype=np.float64)
    filtered_data = convolve2d(data, sum_filter, mode='same', boundary='fill')
    center = np.unravel_index(np.argmax(filtered_data), data.shape)

    max_value = np.max(data)
    threshold = max_value * 0.135
    mask = data >= threshold
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask, intensity_image=data)
    region = next((r for r in regions if r.label == labeled_mask[center]), None)
    if region is None:
        raise ValueError("No region found containing the maximum value")
    y0, x0 = region.centroid
    # regionprops measures from the row axis; plotting/integration use the x axis.
    orientation = np.pi / 2 - region.orientation
    major_axis_radius = region.major_axis_length / 2
    minor_axis_radius = region.minor_axis_length / 2
    ellipticity = major_axis_radius / minor_axis_radius

    # 计算x/y方向半径（主轴、副轴在xy方向的投影）
    # x方向半径 = sqrt((major*cosθ)^2 + (minor*sinθ)^2)
    # y方向半径 = sqrt((major*sinθ)^2 + (minor*cosθ)^2)
    x_dir_radius = np.sqrt((major_axis_radius * np.cos(orientation))**2 +
                           (minor_axis_radius * np.sin(orientation))**2)
    y_dir_radius = np.sqrt((major_axis_radius * np.sin(orientation))**2 +
                           (minor_axis_radius * np.cos(orientation))**2)

    return x_dir_radius, y_dir_radius, (x0, y0), orientation, ellipticity, major_axis_radius, minor_axis_radius


def fit_gaussian_center(data, initial_center, major_radius, minor_radius, orientation):
    """Fit an elliptical 2D Gaussian near the selected spot and return its center."""
    rows, columns = data.shape
    initial_x, initial_y = initial_center
    half_width = max(8, int(np.ceil(3 * major_radius)))
    x_start = max(0, int(np.floor(initial_x - half_width)))
    x_stop = min(columns, int(np.ceil(initial_x + half_width)) + 1)
    y_start = max(0, int(np.floor(initial_y - half_width)))
    y_stop = min(rows, int(np.ceil(initial_y + half_width)) + 1)
    crop = data[y_start:y_stop, x_start:x_stop]
    stride = max(1, int(np.ceil(max(crop.shape) / 128)))
    sample = crop[::stride, ::stride].astype(float)
    peak = float(np.max(sample))
    if peak <= 0:
        raise ValueError('Gaussian center fit requires a positive spot intensity.')
    y_grid, x_grid = np.mgrid[y_start:y_stop:stride, x_start:x_stop:stride]
    values = sample / peak
    background = float(np.percentile(values, 10))
    amplitude = max(float(np.max(values)) - background, 0.01)
    center_bound = max(float(major_radius), 2.0)
    lower = [max(x_start, initial_x - center_bound),
             max(y_start, initial_y - center_bound),
             0.5, 0.5, -np.pi, 0.0, 0.0]
    upper = [min(x_stop - 1, initial_x + center_bound),
             min(y_stop - 1, initial_y + center_bound),
             max(crop.shape), max(crop.shape), np.pi, 2.0, 1.0]
    guess = [initial_x, initial_y, max(major_radius / 2, 0.5),
             max(minor_radius / 2, 0.5), orientation, amplitude, background]

    def residual(parameters):
        x_center, y_center, sigma_major, sigma_minor, angle, height, floor = parameters
        dx = x_grid - x_center
        dy = y_grid - y_center
        major_coordinate = dx * np.cos(angle) + dy * np.sin(angle)
        minor_coordinate = -dx * np.sin(angle) + dy * np.cos(angle)
        model = floor + height * np.exp(-0.5 * (
            (major_coordinate / sigma_major) ** 2
            + (minor_coordinate / sigma_minor) ** 2))
        return (model - values).ravel()

    fit = least_squares(residual, guess, bounds=(lower, upper),
                        loss='soft_l1', f_scale=0.05, max_nfev=60)
    if not fit.success or not np.all(np.isfinite(fit.x[:2])):
        raise ValueError('Could not determine the Gaussian spot center.')
    return float(fit.x[0]), float(fit.x[1])


def fit_fwhm_ellipse(data, center, search_radius):
    """Fit the measured half-height contour around the Gaussian center.

    The ellipse center is held at the Gaussian-fit center. The overlap score is
    intersection / union of the measured half-height component and ellipse.
    """
    x_center, y_center = center
    margin = max(8, int(np.ceil(2 * search_radius)))
    x_start = max(0, int(np.floor(x_center)) - margin)
    x_stop = min(data.shape[1], int(np.ceil(x_center)) + margin + 1)
    y_start = max(0, int(np.floor(y_center)) - margin)
    y_stop = min(data.shape[0], int(np.ceil(y_center)) + margin + 1)
    crop = data[y_start:y_stop, x_start:x_stop]
    background = float(np.percentile(data, 5))
    half_max = background + (float(np.max(crop)) - background) / 2
    if half_max <= background:
        raise ValueError('Cannot fit an FWHM ellipse to a zero-intensity spot.')

    contours = find_contours(crop, half_max)
    local_center = (x_center - x_start, y_center - y_start)
    enclosing = [c for c in contours if len(c) >= 12 and
                 np.linalg.norm(c[0] - c[-1]) <= 1.5 and
                 MplPath(c[:, ::-1]).contains_point(local_center)]
    if not enclosing:
        raise ValueError('No closed FWHM contour encloses the Gaussian center.')
    contour = max(enclosing, key=len)
    points = contour[:, ::-1] - local_center
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_guess, minor_guess = np.sqrt(2 * np.maximum(eigenvalues[::-1], 0.25))
    angle_guess = float(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

    def residual(parameters):
        major, minor, angle = parameters
        u = points[:, 0] * np.cos(angle) + points[:, 1] * np.sin(angle)
        v = -points[:, 0] * np.sin(angle) + points[:, 1] * np.cos(angle)
        return np.sqrt((u / major) ** 2 + (v / minor) ** 2) - 1

    upper = max(crop.shape) * 2.0
    fitted = least_squares(residual, [major_guess, minor_guess, angle_guess],
                           bounds=([0.5, 0.5, -2*np.pi],
                                   [upper, upper, 2*np.pi]),
                           loss='soft_l1', f_scale=0.05)
    if not fitted.success:
        raise ValueError('Could not fit the FWHM contour ellipse.')
    major, minor, angle = fitted.x
    if minor > major:
        major, minor = minor, major
        angle += np.pi / 2
    angle = (angle + np.pi / 2) % np.pi - np.pi / 2
    x_half_radius = 1 / np.sqrt((np.cos(angle) / major) ** 2 +
                                (np.sin(angle) / minor) ** 2)
    y_half_radius = 1 / np.sqrt((np.sin(angle) / major) ** 2 +
                                (np.cos(angle) / minor) ** 2)

    measured = crop >= half_max
    components, _ = ndimage_label(measured)
    center_row = int(np.clip(np.rint(local_center[1]), 0, crop.shape[0]-1))
    center_col = int(np.clip(np.rint(local_center[0]), 0, crop.shape[1]-1))
    component_id = components[center_row, center_col]
    if component_id == 0:
        raise ValueError('Gaussian center lies outside the FWHM region.')
    measured = components == component_id
    yy, xx = np.ogrid[y_start:y_stop, x_start:x_stop]
    u, v = calculate_rotated_coordinates(x_center, y_center, xx, yy, angle)
    ellipse = (u / major) ** 2 + (v / minor) ** 2 <= 1
    overlap = np.count_nonzero(measured & ellipse) / np.count_nonzero(measured | ellipse)
    return dict(major=major, minor=minor, angle=angle,
                fwhm_x=2*x_half_radius, fwhm_y=2*y_half_radius,
                half_max=half_max, overlap=overlap)

def calculate_radius_threshold(data):
    data = preprocess_data(data)
    max_value = np.max(data)
    center = np.unravel_index(np.argmax(data), data.shape)
    y_center, x_center = center

    # Calculate threshold in x direction
    x_profile = data[y_center, :]
    threshold = max_value * 0.135
    x_indices = np.where(x_profile >= threshold)[0]
    x_radius = (x_indices[-1] - x_indices[0]) / 2

    # Calculate threshold in y direction
    y_profile = data[:, x_center]
    y_indices = np.where(y_profile >= threshold)[0]
    y_radius = (y_indices[-1] - y_indices[0]) / 2

    major_axis_radius = max(x_radius, y_radius)
    minor_axis_radius = min(x_radius, y_radius)

    # Calculate ellipticity
    ellipticity = major_axis_radius / minor_axis_radius

    # Orientation is zero for a regular ellipse
    orientation = 0 if x_radius >= y_radius else np.pi / 2

    return x_radius, y_radius, (x_center, y_center), orientation, ellipticity, major_axis_radius, minor_axis_radius

def calculate_radius_fwhm(data):
    data = preprocess_data(data)
    max_value = np.max(data)
    center = np.unravel_index(np.argmax(data), data.shape)
    y_center, x_center = center

    # Calculate FWHM in x direction
    x_profile = data[y_center, :]
    half_max = max_value / 2
    x_indices = np.where(x_profile >= half_max)[0]
    x_fwhm = x_indices[-1] - x_indices[0]

    # Calculate FWHM in y direction
    y_profile = data[:, x_center]
    y_indices = np.where(y_profile >= half_max)[0]
    y_fwhm = y_indices[-1] - y_indices[0]

    # Calculate radii and ellipticity
    x_radius = x_fwhm / 2
    y_radius = y_fwhm / 2
    major_axis_radius = max(x_radius, y_radius)
    minor_axis_radius = min(x_radius, y_radius)

    # Calculate ellipticity
    ellipticity = major_axis_radius / minor_axis_radius

    # Orientation is zero for a regular ellipse
    orientation = 0 if x_radius >= y_radius else np.pi / 2

    return x_radius, y_radius, (x_center, y_center), orientation, ellipticity, major_axis_radius, minor_axis_radius


def plot_ellipse_on_image(major_axis_radius, minor_axis_radius, center, orientation, data):
    global region_total, PIXEL_WIDTH_MICRONS

    # 获取数据形状
    rows, cols = data.shape

    # 椭圆中心（像素坐标）
    center_x_pixel, center_y_pixel = center

    # 将半径转换为微米
    major_axis_um = major_axis_radius * PIXEL_WIDTH_MICRONS
    minor_axis_um = minor_axis_radius * PIXEL_WIDTH_MICRONS

    # 创建一个以椭圆中心为原点，微米为单位的坐标系统
    # 计算图像边界相对于椭圆中心的微米坐标
    left_um = -center_x_pixel * PIXEL_WIDTH_MICRONS
    right_um = (cols - center_x_pixel) * PIXEL_WIDTH_MICRONS
    bottom_um = -(rows - center_y_pixel) * PIXEL_WIDTH_MICRONS  # 注意：y轴方向
    top_um = center_y_pixel * PIXEL_WIDTH_MICRONS

    fig, ax = plt.subplots(figsize=(6.74, 5))

    # 使用extent参数设置坐标轴范围
    extent = [left_um, right_um, bottom_um, top_um]

    # 显示图像
    cax = ax.imshow(data, cmap='rainbow', aspect='auto',
                    extent=extent, origin='upper')

    # 添加colorbar
    cbar = fig.colorbar(cax, ax=ax, label='Intensity (counts)', pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.label.set_fontsize(12)

    # 绘制椭圆（现在原点就是椭圆中心，即(0,0)点）
    theta = np.linspace(0, 2 * np.pi, 100)

    # 绘制1/e²椭圆
    x_ellipse = major_axis_um * np.cos(theta)
    y_ellipse = minor_axis_um * np.sin(theta)

    # 应用旋转
    x_rot = x_ellipse * np.cos(orientation) - y_ellipse * np.sin(orientation)
    y_rot = x_ellipse * np.sin(orientation) + y_ellipse * np.cos(orientation)

    # 绘制椭圆（中心在(0,0)）
    ax.plot(x_rot, y_rot, color=rgb2, linewidth=1.5,
            label=r'$\frac{1}{e^2}$ ellipse', linestyle='--')

    # 绘制中心点（现在就是(0,0)点）
    ax.plot(0, 0, color='white', marker='x', markersize=10,
            markeredgewidth=1)

    # 绘制区域椭圆（region_total倍）
    x_total = region_total * major_axis_um * np.cos(theta)
    y_total = region_total * minor_axis_um * np.sin(theta)
    x_total_rot = x_total * np.cos(orientation) - y_total * np.sin(orientation)
    y_total_rot = x_total * np.sin(orientation) + y_total * np.cos(orientation)
    ax.plot(x_total_rot, y_total_rot, color=rgb4, linestyle='--',
            linewidth=1.5, label='total energy ellipse')

    # 计算并显示积分比例
    ratio = calculate_integral_ratio(data, major_axis_radius, minor_axis_radius, center, orientation, region_total)


    x_dir_radius = np.sqrt((major_axis_radius * np.cos(orientation))**2 +
                           (minor_axis_radius * np.sin(orientation))**2)
    y_dir_radius = np.sqrt((major_axis_radius * np.sin(orientation))**2 +
                           (minor_axis_radius * np.cos(orientation))**2)

    # 添加文本框显示参数
    text_str = (
    f'$a$ = {major_axis_um:.2f} μm\n'
    f'$b$ = {minor_axis_um:.2f} μm\n'
    f'$w_x$ = {x_dir_radius * PIXEL_WIDTH_MICRONS:.2f} μm\n'
    f'$w_y$ = {y_dir_radius * PIXEL_WIDTH_MICRONS:.2f} μm\n'
    f'ε = {major_axis_radius/minor_axis_radius:.2f}\n'
    f'η = {ratio:.3f}\n'
    f'θ = {orientation/np.pi*180:.2f}°'
    )
    ax.text(0.05, 0.955, text_str, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=1, edgecolor='k'),
            fontsize=12, linespacing=1.5)

    max_radius_um = max(major_axis_um, minor_axis_um)
    region_range_x = (region_total + 1) * max_radius_um
    region_range_y = (region_total + 1) * max_radius_um

    ax.set_xlim(-region_range_x, region_range_x)
    ax.set_ylim(-region_range_y, region_range_y)

    ax.set_xlabel('x (μm)', fontsize=12)
    ax.set_ylabel('y (μm)', fontsize=12)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', which='major', direction='in',
                   width=1.2, size=6, labelsize=12)
    ax.legend(loc='upper right', fontsize=12, framealpha=1)
    ax.set_aspect('equal', adjustable='box')

    # 保存图像
    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / 'figure'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'laser_spot.png', dpi=300, bbox_inches='tight')
    plt.show()

    return ratio

def calculate_integral_ratio(data, x_radius, y_radius, center, orientation, region_total):
    y_indices, x_indices = np.ogrid[:data.shape[0], :data.shape[1]]
    x_center, y_center = center
    x_rot, y_rot = calculate_rotated_coordinates(x_center, y_center, x_indices, y_indices, orientation)
    mask = (x_rot / x_radius)**2 + (y_rot / y_radius)**2 <= 1
    total_mask = (x_rot / (region_total * x_radius))**2 + (y_rot / (region_total * y_radius))**2 <= 1
    total_integral = np.sum(data[total_mask])
    selected_integral = np.sum(data[mask])
    ratio = selected_integral / total_integral
    return ratio

def scan_region_total(data, x_radius, y_radius, center, orientation):
    ratios = []
    region_totals = np.arange(1, 10, 0.5)
    for region_total in region_totals:
        ratio = calculate_integral_ratio(data, x_radius, y_radius, center, orientation, region_total)
        ratios.append(ratio)
    plt.figure()
    plt.plot(region_totals, ratios, marker='o')
    plt.xlabel(r'region size [$1/e^{2}~w_0$]')
    plt.ylabel('Ratio')
    plt.grid(True)
    plt.show()

def scan_energy_ratio(data, x_radius, y_radius, center, orientation):
    ratios = []
    multiples = np.arange(0.1, 10, 0.5)
    for multiple in multiples:
        ratio = calculate_integral_ratio(data, x_radius, y_radius, center, orientation, multiple)
        ratios.append(ratio)
    plt.figure()
    plt.plot(multiples, ratios, marker='o')
    plt.xlabel(r'Multiples of $1/e^{2}$ radius')
    plt.ylabel('Energy Ratio')
    plt.grid(True)
    plt.show()

def cal_a0_pic(data, x_radius, y_radius, center, orientation):
    global energy, duration, dx, dy, lambda_L, region_total
    x_center, y_center = center
    peak_x = int(np.clip(np.rint(x_center), 0, data.shape[1] - 1))
    peak_y = int(np.clip(np.rint(y_center), 0, data.shape[0] - 1))
    peak_counts = np.mean(data[max(0, peak_y-1):min(data.shape[0], peak_y+2),
                               max(0, peak_x-1):min(data.shape[1], peak_x+2)])
    y_indices, x_indices = np.ogrid[:data.shape[0], :data.shape[1]]
    x_rot, y_rot = calculate_rotated_coordinates(x_center, y_center, x_indices, y_indices, orientation)
    total_mask = (x_rot / (region_total * x_radius))**2 + (y_rot / (region_total * y_radius))**2 <= 1
    total_integral = np.sum(data[total_mask])
    I = peak_counts / total_integral * energy / duration / (dx*dy)
    print(f'I: {I:.2e}')
    a0 = np.sqrt(I*(lambda_L*1e6)**2/1.37e18)
    return a0

def cal_a0_gauss(data, x_radius, y_radius, center, orientation):
    global energy, duration, dx, dy, lambda_L, region_total

    # 计算能量集中度
    ratio = calculate_integral_ratio(data, x_radius, y_radius, center, orientation, region_total)

    # 计算高斯光束的面积 (π * x_radius * y_radius)
    beam_area = np.pi * x_radius * y_radius * (dx * dy)

    # 计算激光强度 I = 能量集中度 * 总能量 / (脉冲持续时间 * 光束面积)
    I = 2 * ratio * energy / (duration * beam_area)

    # 根据高斯光束公式计算 a0
    a0 = np.sqrt(I * (lambda_L * 1e6)**2 / 1.37e18)

    print(f'Gaussian Laser Intensity (I): {I:.2e}')
    print(f'a0 (Gaussian): {a0:.2f}')

    return a0


def main(file_path):
    data = load_spot_data(file_path)
    data = preprocess_data(data)

    x_radius, y_radius, center, orientation, ellipticity, major_axis_radius, minor_axis_radius = calculate_radius(data)
    plot_ellipse_on_image(major_axis_radius, minor_axis_radius, center, orientation, data)

    a0 = cal_a0_gauss(data, major_axis_radius, minor_axis_radius, center, orientation)
    # a0 = cal_a0_pic(data, x_radius, y_radius, center, orientation)
    print(f'X radius: {x_radius * PIXEL_WIDTH_MICRONS:.2f} microns')
    print(f'Y radius: {y_radius * PIXEL_WIDTH_MICRONS:.2f} microns')
    print(f'Major axis radius: {major_axis_radius * PIXEL_WIDTH_MICRONS:.2f} microns')
    print(f'Minor axis radius: {minor_axis_radius * PIXEL_WIDTH_MICRONS:.2f} microns')
    print(f'Ellipticity: {ellipticity:.2f}')
    print(f'a0: {a0:.2f}')
    print(f'angle: {orientation/np.pi*180:.2f} °')
    scan_region_total(data, major_axis_radius, minor_axis_radius, center, orientation)
    scan_energy_ratio(data, major_axis_radius, minor_axis_radius, center, orientation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze a CSV or grayscale TIFF laser spot image.')
    parser.add_argument('file', help='Path to a CSV, TIF, or TIFF file')
    main(parser.parse_args().file)
