import argparse
from pathlib import Path
import numpy as np
import analysis  # Uses functions from analysis.py
import matplotlib.pyplot as plt
from data_io import SUPPORTED_SUFFIXES, load_spot_data

# 设置全局样式（与plot_ellipse_on_image保持一致）
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

parser = argparse.ArgumentParser(description='Analyze a folder of CSV and grayscale TIFF spot images.')
parser.add_argument('folder', help='Folder containing spot images')
data_folder = Path(parser.parse_args().folder)
output_dir = Path(__file__).resolve().parent / 'figure'
output_dir.mkdir(exist_ok=True)
file_paths = sorted(path for path in data_folder.iterdir()
                    if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
if not file_paths:
    raise FileNotFoundError(f'No CSV or TIFF files found in {data_folder}')

# Initialize arrays to store results
a0_list, x_radius_list, y_radius_list, ratio_list = [], [], [], []
center_x_list, center_y_list, file_names = [], [], []

# Use the region_total defined in analysis or override here
region_total = analysis.region_total

# Process each spot image in the folder
for file_path in file_paths:
    try:
        data = analysis.preprocess_data(load_spot_data(file_path))
        
        # Calculate radius and other parameters
        x_radius, y_radius, center, orientation, _, major_axis_radius, minor_axis_radius = analysis.calculate_radius(data)
        x_radius_plot = x_radius * analysis.PIXEL_WIDTH_MICRONS
        y_radius_plot = y_radius * analysis.PIXEL_WIDTH_MICRONS
        a0 = analysis.cal_a0_gauss(data, major_axis_radius, minor_axis_radius, center, orientation)
        ratio = analysis.calculate_integral_ratio(data, major_axis_radius, minor_axis_radius, center, orientation, region_total)
        
        # 记录中心位置（像素坐标）
        center_x, center_y = center
        
        # Append results
        a0_list.append(a0)
        x_radius_list.append(x_radius_plot)
        y_radius_list.append(y_radius_plot)
        ratio_list.append(ratio)
        center_x_list.append(center_x)
        center_y_list.append(center_y)
        file_names.append(file_path.name)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert lists into arrays and print results
if not file_names:
    raise ValueError('No spot images could be analyzed successfully')
print("Files:", np.array(file_names))
print("a0 values:", np.array(a0_list))
print("x_radius values:", np.array(x_radius_list))
print("y_radius values:", np.array(y_radius_list))
print("Energy Concentration Ratios:", np.array(ratio_list))

# 计算指向抖动（1σ）
def calculate_pointing_jitter(center_x_list, center_y_list):
    """计算激光指向抖动（1σ）"""
    # 将像素坐标转换为微米
    center_x_um = np.array(center_x_list) * analysis.PIXEL_WIDTH_MICRONS
    center_y_um = np.array(center_y_list) * analysis.PIXEL_WIDTH_MICRONS
    
    # 计算中心点的均值
    mean_x = np.mean(center_x_um)
    mean_y = np.mean(center_y_um)
    
    # 计算偏移量
    offsets_x = center_x_um - mean_x
    offsets_y = center_y_um - mean_y
    
    # 计算标准差（1σ）
    sigma_x = np.std(offsets_x, ddof=1)  # 样本标准差
    sigma_y = np.std(offsets_y, ddof=1)
    
    # 计算径向偏移（点到中心的距离）
    radial_offsets = np.sqrt(offsets_x**2 + offsets_y**2)
    
    # 计算径向RMS
    sigma_r = np.std(radial_offsets, ddof=1)  # 样本标准差
    
    return {
        'mean_x': mean_x,
        'mean_y': mean_y,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'sigma_r': sigma_r,
        'offsets_x': offsets_x,
        'offsets_y': offsets_y,
        'radial_offsets': radial_offsets
    }

# 计算抖动参数
jitter_params = calculate_pointing_jitter(center_x_list, center_y_list)

# Create scatter plot for x_radius vs y_radius with consistent styling
def plot_radius_scatter(x_radius_list, y_radius_list):
    fig, ax = plt.subplots(figsize=(6.74, 5))
    
    # 绘制散点图
    scatter = ax.scatter(x_radius_list, y_radius_list, marker='o', s=60, 
                         color=rgb1, edgecolor='black', linewidth=0.5, alpha=0.8, label=f'Exp Points (N={len(x_radius_list)})')
    
    # 计算平均值和标准差
    x_mean = np.mean(x_radius_list)
    y_mean = np.mean(y_radius_list)
    x_std = np.std(x_radius_list, ddof=1)
    y_std = np.std(y_radius_list, ddof=1)
    # 添加均值点
    ax.scatter(x_mean, y_mean, marker='s', s=60, 
               color=rgb3, edgecolor='black', linewidth=1.5,
               label=f'Mean: ({x_mean:.1f}, {y_mean:.1f})')

    # 添加误差椭圆
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(xy=(x_mean, y_mean), width=2*x_std, height=2*y_std,
                      edgecolor=rgb3, facecolor='none', linestyle='--',
                      linewidth=1, alpha=0.7, label=r'1$\sigma$ ellipse')
    ax.add_patch(ellipse)

    # 添加对角线参考线（x=y）
    max_val = max(max(x_radius_list), max(y_radius_list)) * 1.1
    min_val = min(min(x_radius_list), min(y_radius_list)) * 0.9
    ax.plot([min_val, max_val], [min_val, max_val], 
            color='gray', linestyle='--', linewidth=1, alpha=0.6, 
            label='x = y')
    

    
    # 设置坐标轴
    ax.set_xlabel(r'$x_{r}~[\mu m]$', fontsize=12)
    ax.set_ylabel(r'$y_{r}~[\mu m]$', fontsize=12)
    
    # 设置坐标轴范围
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置刻度
    ax.tick_params(axis='both', which='major', direction='in', 
                   width=1.2, size=6, labelsize=12)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=12, framealpha=1)
    
    # 添加文本框显示统计数据
    text_str = (f'$\mu_x$ = {x_mean:.1f} μm\n'
                f'$\mu_y$ = {y_mean:.1f} μm\n'
                f'$\sigma_x$ = {x_std:.1f} μm\n'
                f'$\sigma_y$ = {y_std:.1f} μm'
                )
    
    ax.text(0.95, 0.05, text_str, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=1, 
                     edgecolor='k'),
            fontsize=12, linespacing=1.5)
    
    # 确保长宽比正确
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radius_scatter.png',
                dpi=300, bbox_inches='tight')
    plt.show()

# Create scatter plot for a0 vs Energy Ratio with consistent styling
def plot_a0_ratio_scatter(a0_list, ratio_list):
    fig, ax = plt.subplots(figsize=(6.74, 5))
    
    # 绘制散点图
    scatter = ax.scatter(a0_list, ratio_list, marker='o', s=60, 
                         color=rgb2, edgecolor='black', linewidth=0.5, alpha=0.8, label=f'Exp Points (N={len(a0_list)})')
    
    # 计算平均值和标准差
    a0_mean = np.mean(a0_list)
    ratio_mean = np.mean(ratio_list)
    a0_std = np.std(a0_list, ddof=1)
    ratio_std = np.std(ratio_list, ddof=1)
    
    # 添加均值点与误差条
    ax.errorbar(a0_mean, ratio_mean, xerr=a0_std, yerr=ratio_std,
                fmt='s', markersize=7, ecolor=rgb3, mec='black', mfc=rgb3,
                capsize=4, elinewidth=1.2, markeredgewidth=1.2,
                label=f'Mean: ({a0_mean:.2f}±{a0_std:.2f}, {ratio_mean:.3f}±{ratio_std:.3f})')
    
    # 设置坐标轴
    ax.set_xlabel(r'$a_0$', fontsize=12)
    ax.set_ylabel('Energy Concentration Ratio η', fontsize=12)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置刻度
    ax.tick_params(axis='both', which='major', direction='in', 
                   width=1.2, size=6, labelsize=12)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=12, framealpha=1)
    
    # 添加文本框显示统计数据
    text_str = (f'$\mu_{{a_0}}$ = {a0_mean:.2f}\n'
                f'$\mu_{{\eta}}$ = {ratio_mean:.3f}\n'
                f'$\sigma_{{a_0}}$ = {a0_std:.2f}\n'
                f'$\sigma_{{\eta}}$ = {ratio_std:.3f}')
    
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=1, 
                     edgecolor='k'),
            fontsize=12, linespacing=1.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'a0_ratio_scatter.png',
                dpi=300, bbox_inches='tight')
    plt.show()

# 创建指向抖动散点图
def plot_pointing_jitter(jitter_params):
    fig, ax = plt.subplots(figsize=(6.74, 5))
    
    # 获取数据
    offsets_x = jitter_params['offsets_x']
    offsets_y = jitter_params['offsets_y']
    sigma_x = jitter_params['sigma_x']
    sigma_y = jitter_params['sigma_y']
    sigma_r = jitter_params['sigma_r']
    
    # 绘制中心点散点图
    scatter = ax.scatter(offsets_x, offsets_y, marker='o', s=60,
                         color=rgb4, edgecolor='black', linewidth=0.5, alpha=0.8,
                         label=f'Beam centers (N={len(offsets_x)})')
    
    # 绘制原点（平均位置）
    ax.scatter(0, 0, marker='+', s=100, color='black', linewidth=2,
               label='Mean position')
    
    # 绘制1σ椭圆
    from matplotlib.patches import Ellipse
    ellipse_1sigma = Ellipse(xy=(0, 0), width=2*sigma_x, height=2*sigma_y,
                             edgecolor=rgb3, facecolor='none', linestyle='-',
                             linewidth=1.5, alpha=0.8, label='1σ ellipse')
    ax.add_patch(ellipse_1sigma)
    
    # 绘制1σ圆（RMS径向）
    circle_rms = plt.Circle((0, 0), sigma_r, edgecolor=rgb1, facecolor='none', 
                           linestyle='--', linewidth=1.5, alpha=0.8, label=f'1σ circle')
    ax.add_patch(circle_rms)
    
    # 计算坐标轴范围
    max_offset = max(np.max(np.abs(offsets_x)), np.max(np.abs(offsets_y)), sigma_r) * 1.2
    ax.set_xlim(-max_offset, max_offset)
    ax.set_ylim(-max_offset, max_offset)
    
    # 设置坐标轴标签
    ax.set_xlabel('x (μm)', fontsize=12)
    ax.set_ylabel('y (μm)', fontsize=12)
    ax.set_title('Pointing Jitter', fontsize=14, pad=10)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置刻度
    ax.tick_params(axis='both', which='major', direction='in', 
                   width=1.2, size=6, labelsize=12)
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=12, framealpha=1)
    
    # 添加文本框显示统计数据
    text_str = (f'Pointing jitter (1σ):\n'
                f'  $σ_x$ = {sigma_x:.2f} μm\n'
                f'  $σ_y$ = {sigma_y:.2f} μm\n'
                f'  $σ_r$ = {sigma_r:.2f} μm')
    
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=1, 
                     edgecolor='k'),
            fontsize=12, linespacing=1.5)
    
    # 确保长宽比正确
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pointing_jitter.png',
                dpi=300, bbox_inches='tight')
    plt.show()

# 调用绘图函数
plot_radius_scatter(x_radius_list, y_radius_list)
plot_a0_ratio_scatter(a0_list, ratio_list)
plot_pointing_jitter(jitter_params)

# Print average and standard deviation
print("\nStatistical Summary:")
print(f"a0 mean: {np.mean(a0_list):.2f}, std: {np.std(a0_list, ddof=1):.2f}")
print(f"x_radius mean: {np.mean(x_radius_list):.2f} μm, std: {np.std(x_radius_list, ddof=1):.2f} μm")
print(f"y_radius mean: {np.mean(y_radius_list):.2f} μm, std: {np.std(y_radius_list, ddof=1):.2f} μm")
print(f"ratio mean: {np.mean(ratio_list):.3f}, std: {np.std(ratio_list, ddof=1):.3f}")

# 打印指向抖动结果
print("\nPointing Jitter Analysis (1σ):")
print(f"Mean position: X = {jitter_params['mean_x']:.1f} μm, Y = {jitter_params['mean_y']:.1f} μm")
print(f"X-direction jitter: σ_x = {jitter_params['sigma_x']:.2f} μm")
print(f"Y-direction jitter: σ_y = {jitter_params['sigma_y']:.2f} μm")
