from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import analysis
from data_io import SUPPORTED_SUFFIXES, load_spot_data

CANVAS = '#EDF3F5'
SURFACE = '#FFFFFF'
INK = '#153240'
MUTED = '#5E7480'
ACCENT = '#117E9C'
RULE = '#CCDDE3'

SINGLE_COLUMN_SIZE = (3.375, 2.35)
FIGURE_ASPECT = SINGLE_COLUMN_SIZE[0] / SINGLE_COLUMN_SIZE[1]


def natural_path_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', Path(path).name)]
mpl.rcParams.update({
    'font.family': ['Arial', 'GraphikNaturel-Regular', 'Times New Roman', 'DejaVu Sans'],
    'font.size': 10,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'mathtext.sf': 'Arial',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,
    'axes.formatter.use_mathtext': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': False,
    'ytick.right': False,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 0.8,
    'lines.markersize': 4,
    'axes.grid': False,
})


def single_column_figure(constrained_layout=True):
    fig = Figure(figsize=SINGLE_COLUMN_SIZE, dpi=100,
                 constrained_layout=constrained_layout)
    ax = fig.add_subplot()
    return fig, ax


def add_intensity_colorbar(fig, ax, image):
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes('right', size='4%', pad=0.08)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label('Intensity (counts)', fontsize=9)
    colorbar.ax.tick_params(direction='in', labelsize=9, width=0.5, length=3)
    colorbar.outline.set_linewidth(0.5)


def spot_image_extent(shape, center, pixel_width_microns):
    """Place the fitted spot center at the physical (0, 0) coordinates."""
    rows, columns = shape
    x_center, y_center = center
    return ((-0.5 - x_center) * pixel_width_microns,
            (columns - 0.5 - x_center) * pixel_width_microns,
            (rows - 0.5 - y_center) * pixel_width_microns,
            (-0.5 - y_center) * pixel_width_microns)


def label_micron_axes(ax):
    ax.set_xlabel(r'$x$ (μm)')
    ax.set_ylabel(r'$y$ (μm)')


def plot_fwhm_ellipse(ax, fitted, pixel_width_microns):
    theta = np.linspace(0, 2*np.pi, 200)
    major = fitted['major']
    minor = fitted['minor']
    angle = fitted['angle']
    x_fit = major*np.cos(theta)*np.cos(angle) - minor*np.sin(theta)*np.sin(angle)
    y_fit = major*np.cos(theta)*np.sin(angle) + minor*np.sin(theta)*np.cos(angle)
    ax.plot(x_fit*pixel_width_microns, y_fit*pixel_width_microns,
            color='white', linestyle=':', linewidth=1.1,
            path_effects=[path_effects.Stroke(linewidth=1.9, foreground='black'),
                          path_effects.Normal()],
            label='FWHM ellipse fit')

# Create figure for ellipse overlay
def create_ellipse_figure(x_radius, y_radius, center, orientation, data, region_total,
                          pixel_width_microns=None, fwhm_ellipse=None):
    if pixel_width_microns is None:
        pixel_width_microns = analysis.PIXEL_WIDTH_MICRONS
    fig, ax = single_column_figure(constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.75, bottom=0.20, top=0.94)
    cax = ax.imshow(data, cmap='jet', aspect='auto',
                    norm=Normalize(vmin=0, vmax=float(np.max(data))),
                    extent=spot_image_extent(data.shape, center, pixel_width_microns))
    add_intensity_colorbar(fig, ax, cax)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(0, 0, color='white', marker='x', markersize=5, linestyle='none')
    if fwhm_ellipse is not None:
        plot_fwhm_ellipse(ax, fwhm_ellipse, pixel_width_microns)
    x_total = region_total * x_radius * np.cos(theta)
    y_total = region_total * y_radius * np.sin(theta)
    x_total_rot = x_total * np.cos(orientation) - y_total * np.sin(orientation)
    y_total_rot = x_total * np.sin(orientation) + y_total * np.cos(orientation)
    ax.plot(x_total_rot * pixel_width_microns, y_total_rot * pixel_width_microns,
            color='#ff7f0e', linestyle='--', linewidth=0.8,
            label='Integration region')
    ax.set_xlim(-(region_total+1)*x_radius*pixel_width_microns,
                (region_total+1)*x_radius*pixel_width_microns)
    ax.set_ylim(-(region_total+1)*y_radius*pixel_width_microns,
                (region_total+1)*y_radius*pixel_width_microns)
    label_micron_axes(ax)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8, facecolor='white', edgecolor='white')
    return fig

# NEW: Create a zoomed ellipse overlay figure (with restricted axis limits)
def create_ellipse_zoom_figure(x_radius, y_radius, center, orientation, data, region_total,
                               pixel_width_microns=None, fwhm_ellipse=None):
    if pixel_width_microns is None:
        pixel_width_microns = analysis.PIXEL_WIDTH_MICRONS
    fig, ax = single_column_figure(constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.75, bottom=0.20, top=0.94)
    cax = ax.imshow(data, cmap='jet', aspect='auto',
                    norm=Normalize(vmin=0, vmax=float(np.max(data))),
                    extent=spot_image_extent(data.shape, center, pixel_width_microns))
    add_intensity_colorbar(fig, ax, cax)
    ax.plot(0, 0, color='white', marker='x', markersize=5, linestyle='none')
    x_center, y_center = center
    half_width = max(8, int(np.ceil(2 * max(x_radius, y_radius))))
    x_start = max(0, int(np.floor(x_center)) - half_width)
    x_stop = min(data.shape[1], int(np.ceil(x_center)) + half_width + 1)
    y_start = max(0, int(np.floor(y_center)) - half_width)
    y_stop = min(data.shape[0], int(np.ceil(y_center)) + half_width + 1)
    local_data = data[y_start:y_stop, x_start:x_stop]
    background = float(np.percentile(data, 5))
    peak = float(np.max(local_data))
    fwhm_level = (fwhm_ellipse['half_max'] if fwhm_ellipse is not None else
                  background + (peak - background) / 2)
    e2_level = background + (peak - background) * np.exp(-2)
    x_um = (np.arange(x_start, x_stop) - x_center) * pixel_width_microns
    y_um = (np.arange(y_start, y_stop) - y_center) * pixel_width_microns
    e2_contour = ax.contour(x_um, y_um, local_data, levels=[e2_level],
                            colors='white', linewidths=0.8)
    fwhm_contour = ax.contour(x_um, y_um, local_data, levels=[fwhm_level],
                              colors='#ff7f0e', linestyles=':', linewidths=1.1)
    contour_effects = [
        path_effects.Stroke(linewidth=1.6, foreground='black'),
        path_effects.Normal()]
    if hasattr(e2_contour, 'set_path_effects'):
        e2_contour.set_path_effects(contour_effects)
    else:
        for collection in e2_contour.collections:
            collection.set_path_effects(contour_effects)
    # Modified axis limits for zoomed view
    ax.set_xlim(-2*x_radius*pixel_width_microns, 2*x_radius*pixel_width_microns)
    ax.set_ylim(-2*y_radius*pixel_width_microns, 2*y_radius*pixel_width_microns)
    label_micron_axes(ax)
    ax.set_aspect('equal')
    legend_handles = [
        Line2D([], [], color='white', linewidth=0.8,
               path_effects=contour_effects, label=r'$1/e^2$ contour'),
        Line2D([], [], color='#ff7f0e', linestyle=':', linewidth=1.1,
               label='FWHM contour')]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
              facecolor='white', edgecolor='white')
    return fig

def calculate_a0_cuts(data, center, pixel_width_microns, a0):
    """Sample horizontal and vertical a0 profiles through the fitted center."""
    x_center, y_center = center
    rows, columns = data.shape
    center_x = int(np.clip(np.rint(x_center), 0, columns - 1))
    center_y = int(np.clip(np.rint(y_center), 0, rows - 1))
    reference = np.mean(data[max(0, center_y-1):min(rows, center_y+2),
                             max(0, center_x-1):min(columns, center_x+2)])
    if reference <= 0:
        raise ValueError('Cannot calculate a0 profiles from zero center intensity.')
    x_pixels = np.unique(np.append(np.arange(columns, dtype=float), x_center))
    y_pixels = np.unique(np.append(np.arange(rows, dtype=float), y_center))
    horizontal = map_coordinates(data, [np.full_like(x_pixels, y_center), x_pixels],
                                 order=1, mode='nearest')
    vertical = map_coordinates(data, [y_pixels, np.full_like(y_pixels, x_center)],
                               order=1, mode='nearest')
    scale = a0 / np.sqrt(reference)
    return ((x_pixels - x_center) * pixel_width_microns,
            scale * np.sqrt(np.maximum(horizontal, 0)),
            (y_pixels - y_center) * pixel_width_microns,
            scale * np.sqrt(np.maximum(vertical, 0)))


def calculate_a0_axis_cuts(data, center, orientation, pixel_width_microns, a0):
    """Sample a0 profiles through the center along the ellipse axes."""
    x_center, y_center = center
    rows, columns = data.shape
    center_x = int(np.clip(np.rint(x_center), 0, columns - 1))
    center_y = int(np.clip(np.rint(y_center), 0, rows - 1))
    reference = np.mean(data[max(0, center_y-1):min(rows, center_y+2),
                             max(0, center_x-1):min(columns, center_x+2)])
    if reference <= 0:
        raise ValueError('Cannot calculate a0 profiles from zero center intensity.')

    def sample_axis(x_direction, y_direction):
        bounds = []
        if abs(x_direction) > 1e-12:
            bounds.append(sorted(((0 - x_center) / x_direction,
                                  (columns - 1 - x_center) / x_direction)))
        if abs(y_direction) > 1e-12:
            bounds.append(sorted(((0 - y_center) / y_direction,
                                  (rows - 1 - y_center) / y_direction)))
        lower = max(bound[0] for bound in bounds)
        upper = min(bound[1] for bound in bounds)
        distances = np.linspace(lower, upper,
                                int(np.ceil(upper - lower)) + 1)
        distances = np.unique(np.append(distances, 0.0))
        values = map_coordinates(
            data,
            [y_center + distances * y_direction,
             x_center + distances * x_direction],
            order=1, mode='nearest')
        scale = a0 / np.sqrt(reference)
        return (distances * pixel_width_microns,
                scale * np.sqrt(np.maximum(values, 0)))

    major = sample_axis(np.cos(orientation), np.sin(orientation))
    minor = sample_axis(-np.sin(orientation), np.cos(orientation))
    return major[0], major[1], minor[0], minor[1]


def create_a0_profile_figure(cuts, x_radius, y_radius, pixel_width_microns):
    x_positions, x_a0, y_positions, y_a0 = cuts
    fig, ax = single_column_figure()
    ax.plot(x_positions, x_a0, color=ACCENT, linewidth=1.2, label='x cut')
    ax.plot(y_positions, y_a0, color='#D17832', linewidth=1.2,
            linestyle='--', label='y cut')
    ax.axvline(0, color=RULE, linewidth=0.8, linestyle=':')
    span = 2 * max(x_radius, y_radius) * pixel_width_microns
    ax.set_xlim(-span, span)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Position along $x$ / $y$ (μm)')
    ax.set_ylabel(r'$a_0$')
    ax.legend(loc='upper right', fontsize=8, facecolor='white', edgecolor='white')
    return fig


def create_a0_axis_profile_figure(cuts, major_radius, minor_radius,
                                  pixel_width_microns):
    major_positions, major_a0, minor_positions, minor_a0 = cuts
    fig, ax = single_column_figure()
    ax.plot(major_positions, major_a0, color=ACCENT, linewidth=1.2,
            label='major-axis cut')
    ax.plot(minor_positions, minor_a0, color='#D17832', linewidth=1.2,
            linestyle='--', label='minor-axis cut')
    ax.axvline(0, color=RULE, linewidth=0.8, linestyle=':')
    span = 2 * max(major_radius, minor_radius) * pixel_width_microns
    ax.set_xlim(-span, span)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Position along major / minor axis (μm)')
    ax.set_ylabel(r'$a_0$')
    ax.legend(loc='upper right', fontsize=8, facecolor='white', edgecolor='white')
    return fig


def apply_analysis_parameters(params):
    pixel_width = params['PIXEL_WIDTH_MICRONS']
    analysis.PIXEL_WIDTH_MICRONS = pixel_width
    analysis.dx = pixel_width * 1e-4
    analysis.dy = pixel_width * 1e-4
    analysis.energy = params['energy']
    analysis.duration = params['duration'] * 1e-15
    analysis.lambda_L = params['lambda_L'] * 1e-9
    analysis.region_total = params['region_total']


def analyze_spot_file(file_path, params, include_profiles=False):
    """Run the GUI analysis pipeline for one spot image."""
    apply_analysis_parameters(params)
    file_path = Path(file_path)
    data = analysis.preprocess_data(load_spot_data(file_path))
    _, _, center, orientation, _, major_radius, minor_radius = analysis.calculate_radius(data)
    center = analysis.fit_gaussian_center(data, center, major_radius,
                                          minor_radius, orientation)
    fwhm = analysis.fit_fwhm_ellipse(data, center, major_radius)
    scale = np.sqrt(2 / np.log(2))
    major_e2 = fwhm['major'] * scale
    minor_e2 = fwhm['minor'] * scale
    x_radius = fwhm['fwhm_x'] * scale / 2
    y_radius = fwhm['fwhm_y'] * scale / 2
    orientation = fwhm['angle']
    ellipticity = fwhm['major'] / fwhm['minor']
    a0 = analysis.cal_a0_pic(data, major_e2, minor_e2, center, orientation)
    ratio = analysis.calculate_integral_ratio(
        data, major_e2, minor_e2, center, orientation,
        params['region_total'])
    result = {
        'mode': 'single',
        'file_path': file_path,
        'data': data,
        'radii': (x_radius, y_radius, center, orientation,
                  ellipticity, major_e2, minor_e2),
        'fwhm': fwhm,
        'a0': a0,
        'ratio': ratio,
        'pixel_width': params['PIXEL_WIDTH_MICRONS'],
        'region_total': params['region_total'],
    }
    if include_profiles:
        result['a0_cuts'] = calculate_a0_cuts(
            data, center, params['PIXEL_WIDTH_MICRONS'], a0)
        result['a0_axis_cuts'] = calculate_a0_axis_cuts(
            data, center, orientation, params['PIXEL_WIDTH_MICRONS'], a0)
    return result


def summarize_batch_results(results, requested_paths, errors, pixel_width_microns):
    """Combine per-file GUI results into batch series and 1-sigma statistics."""
    if not results:
        raise ValueError('No selected spot images could be analyzed successfully.')
    shapes = {result['data'].shape for result in results}
    if len(shapes) != 1:
        shape_text = ', '.join(f'{rows}×{columns}' for rows, columns in sorted(shapes))
        raise ValueError(
            f'Average image requires matching image sizes; found {shape_text}.')

    pixel_width = float(pixel_width_microns)
    a0_values = np.asarray([result['a0'] for result in results], dtype=float)
    wx_values = np.asarray([result['radii'][0] * pixel_width
                            for result in results], dtype=float)
    wy_values = np.asarray([result['radii'][1] * pixel_width
                            for result in results], dtype=float)
    centers_pixel = np.asarray([result['radii'][2] for result in results], dtype=float)
    centers_um = centers_pixel * pixel_width
    mean_center_pixel = np.mean(centers_pixel, axis=0)
    offsets_um = centers_um - np.mean(centers_um, axis=0)

    def sample_std(values):
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return {
        'mode': 'batch',
        'requested_paths': tuple(Path(path) for path in requested_paths),
        'file_paths': tuple(result['file_path'] for result in results),
        'source_path': Path(requested_paths[0]).parent / 'batch',
        'errors': tuple(errors),
        'shot_numbers': np.asarray(
            [result.get('shot_number', index)
             for index, result in enumerate(results, start=1)]),
        'a0_values': a0_values,
        'wx_values': wx_values,
        'wy_values': wy_values,
        'center_offsets_um': offsets_um,
        'mean_data': np.mean(np.stack([result['data'] for result in results]), axis=0),
        'mean_center_pixel': tuple(mean_center_pixel),
        'pixel_width': pixel_width,
        'stats': {
            'a0_mean': float(np.mean(a0_values)),
            'a0_std': sample_std(a0_values),
            'wx_mean': float(np.mean(wx_values)),
            'wx_std': sample_std(wx_values),
            'wy_mean': float(np.mean(wy_values)),
            'wy_std': sample_std(wy_values),
            'jitter_x': sample_std(centers_um[:, 0]),
            'jitter_y': sample_std(centers_um[:, 1]),
            'jitter_rms': float(np.sqrt(np.mean(np.sum(offsets_um**2, axis=1)))),
        },
    }


def padded_limits(values):
    """Return data limits with visible padding above and below the samples."""
    values = np.asarray(values, dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = max(0.08 * span, 0.01 * max(abs(lower), abs(upper), 1.0))
    return lower - padding, upper + padding


def create_batch_a0_figure(shot_numbers, a0_values):
    fig, ax = single_column_figure()
    ax.plot(shot_numbers, a0_values, color=ACCENT, marker='o', markersize=3,
            linewidth=1.1)
    ax.set_xlabel('Shot')
    ax.set_ylabel(r'$a_0$')
    if len(shot_numbers) <= 20:
        ax.set_xticks(shot_numbers)
    ax.set_xlim(np.min(shot_numbers) - 0.5, np.max(shot_numbers) + 0.5)
    ax.set_ylim(*padded_limits(a0_values))
    return fig


def create_batch_waists_figure(shot_numbers, wx_values, wy_values):
    fig, ax = single_column_figure()
    ax.plot(shot_numbers, wx_values, color=ACCENT, marker='o', markersize=3,
            linewidth=1.1, label=r'$w_x$')
    ax.plot(shot_numbers, wy_values, color='#D17832', marker='o', markersize=3,
            linewidth=1.1, linestyle='--', label=r'$w_y$')
    ax.set_xlabel('Shot')
    ax.set_ylabel(r'$1/e^2$ radius (μm)')
    if len(shot_numbers) <= 20:
        ax.set_xticks(shot_numbers)
    ax.set_xlim(np.min(shot_numbers) - 0.5, np.max(shot_numbers) + 0.5)
    ax.set_ylim(*padded_limits(np.concatenate((wx_values, wy_values))))
    ax.legend(loc='upper right', fontsize=8, facecolor='white', edgecolor='white')
    return fig


def create_pointing_jitter_figure(center_offsets_um, shot_numbers=None):
    fig, ax = single_column_figure()
    offsets = np.asarray(center_offsets_um, dtype=float)
    ax.scatter(offsets[:, 0], offsets[:, 1], s=24, color=ACCENT,
               edgecolors='white', linewidths=0.5, zorder=2)
    if shot_numbers is None:
        shot_numbers = np.arange(1, len(offsets) + 1)
    for shot, (x_offset, y_offset) in zip(shot_numbers, offsets):
        ax.annotate(str(shot), (x_offset, y_offset), xytext=(3, 3),
                    textcoords='offset points', fontsize=6, color=INK)
    ax.plot(0, 0, marker='+', color='#D17832', markersize=8,
            markeredgewidth=1.2, linestyle='none', label='mean center')
    ax.axhline(0, color=RULE, linewidth=0.8, linestyle=':')
    ax.axvline(0, color=RULE, linewidth=0.8, linestyle=':')
    ax.set_xlabel(r'$x-\langle x\rangle$ (μm)')
    ax.set_ylabel(r'$y-\langle y\rangle$ (μm)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper right', fontsize=8, facecolor='white', edgecolor='white')
    return fig


def create_average_spot_figure(mean_data, mean_center, pixel_width_microns):
    fig, ax = single_column_figure(constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.75, bottom=0.20, top=0.94)
    image = ax.imshow(
        mean_data, cmap='jet', aspect='auto',
        norm=Normalize(vmin=0, vmax=float(np.max(mean_data))),
        extent=spot_image_extent(mean_data.shape, mean_center,
                                 pixel_width_microns))
    ax.plot(0, 0, color='white', marker='x', markersize=5, linestyle='none')
    label_micron_axes(ax)
    ax.set_aspect('equal')
    add_intensity_colorbar(fig, ax, image)
    return fig


def add_plot(parent, title, figure, row, column):
    panel = tk.Frame(parent, bg=SURFACE, highlightbackground=RULE, highlightthickness=1)
    panel.grid(row=row, column=column, sticky='nsew', padx=6, pady=6)
    tk.Label(panel, text=title, bg=SURFACE, fg=INK,
             font=('Bahnschrift', 12, 'bold'), anchor='w').pack(fill='x', padx=16, pady=(11, 7))
    tk.Frame(panel, bg=RULE, height=1).pack(fill='x', padx=16)
    holder = tk.Frame(panel, bg=SURFACE)
    holder.pack(fill='both', expand=True, padx=8, pady=8)
    toolbar_frame = tk.Frame(holder, bg=SURFACE)
    toolbar_frame.pack(side='bottom', fill='x')
    plot_holder = tk.Frame(holder, bg=SURFACE)
    plot_holder.pack(side='top', fill='both', expand=True)
    canvas = FigureCanvasTkAgg(figure, master=plot_holder)
    canvas.draw()
    figure.set_constrained_layout(False)
    widget = canvas.get_tk_widget()
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    def fit_single_column(event):
        width = max(1, event.width - 8)
        height = max(1, event.height - 8)
        if width / height > FIGURE_ASPECT:
            width = max(1, int(height * FIGURE_ASPECT))
        else:
            height = max(1, int(width / FIGURE_ASPECT))
        widget.place(relx=0.5, rely=0.5, anchor='center', width=width, height=height)

    plot_holder.bind('<Configure>', fit_single_column)
    panel._plot_canvas = canvas
    panel._plot_toolbar = toolbar
    return panel


class FocusAnalysisApp:
    PLOT_EXPORT_NAMES = ('ellipse_full', 'ellipse_detail',
                         'a0_xy_profiles', 'a0_axis_profiles')
    BATCH_PLOT_EXPORT_NAMES = ('a0_by_shot', 'waists_by_shot',
                               'pointing_jitter', 'average_spot')
    SINGLE_METRICS = (
        ('x_radius', 'X 1/e² radius'), ('y_radius', 'Y 1/e² radius'),
        ('fwhm_x', 'X FWHM'), ('fwhm_y', 'Y FWHM'),
        ('ellipticity', 'Ellipticity'), ('angle', 'Ellipse angle'),
        ('major_axis', 'Major axis FWHM'), ('minor_axis', 'Minor axis FWHM'),
        ('overlap', 'Ellipse overlap'), ('ratio', 'Energy ratio'))
    DEFAULT_PARAMS = {
        'PIXEL_WIDTH_MICRONS': '0.44',
        'region_total': '16',
        'energy': '2.66',
        'duration': '35',
        'lambda_L': '800',
    }
    PARAM_LABELS = {
        'PIXEL_WIDTH_MICRONS': ('Pixel size', 'μm'),
        'region_total': ('Integration radius', '×'),
        'energy': ('Pulse energy on target', 'J'),
        'duration': ('Duration', 'fs'),
        'lambda_L': ('Wavelength', 'nm'),
    }

    def __init__(self, root):
        self.root = root
        self.current_file = None
        self.current_paths = ()
        self.current_mode = None
        self.param_entries = {}
        self.result_values = {}
        self.metric_labels = {}
        self.metric_cells = {}
        self.busy = False
        self._task_token = 0
        self._results = Queue()
        self.selected_plot = None
        self._configure_window()
        self.plot_menu = tk.Menu(root, tearoff=False)
        self.plot_menu.add_command(label='Save image as…',
                                   command=self.save_selected_plot)
        self._build_header()
        self._build_body()
        self._build_parameters()
        self._build_footer()
        root.bind('<Control-o>', lambda event: self.select_file())
        root.bind('<F5>', lambda event: self.recalculate())
        root.bind_all('<MouseWheel>', self._scroll_plots, add='+')
        root.bind_all('<MouseWheel>', self._scroll_readout, add='+')

    def _configure_window(self):
        root = self.root
        root.title('Focus analysis')
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        width = min(1440, int(screen_width * 0.9))
        height = min(920, int(screen_height * 0.88))
        root.geometry(f'{width}x{height}')
        root.minsize(min(960, width), min(620, height))
        root.configure(bg=CANVAS)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('Plot.Vertical.TScrollbar', background=RULE, troughcolor=CANVAS,
                        bordercolor=CANVAS, arrowcolor=INK)

    def _button(self, parent, text, command, primary=False):
        background = ACCENT if primary else INK
        active = '#0B667F' if primary else '#294958'
        button = tk.Button(parent, text=text, command=command,
                           font=('Segoe UI', 11, 'bold'), fg=SURFACE,
                           bg=background, activebackground=active,
                           activeforeground=SURFACE, relief='flat', cursor='hand2',
                           padx=22, pady=11, borderwidth=0,
                           disabledforeground='#A8BAC2')
        return button

    def _reticle(self, parent, size, background):
        canvas = tk.Canvas(parent, width=size, height=size, bg=background,
                           highlightthickness=0, borderwidth=0)
        mid = size / 2
        radius = size * 0.27
        canvas.create_oval(mid-radius, mid-radius, mid+radius, mid+radius,
                           outline=ACCENT, width=2)
        canvas.create_line(mid, 3, mid, mid-radius-4, fill=ACCENT, width=2)
        canvas.create_line(mid, mid+radius+4, mid, size-3, fill=ACCENT, width=2)
        canvas.create_line(3, mid, mid-radius-4, mid, fill=ACCENT, width=2)
        canvas.create_line(mid+radius+4, mid, size-3, mid, fill=ACCENT, width=2)
        canvas.create_oval(mid-2, mid-2, mid+2, mid+2, fill=ACCENT, outline=ACCENT)
        return canvas

    def _build_header(self):
        header = tk.Frame(self.root, bg=INK, height=86)
        header.grid(row=0, column=0, sticky='ew')
        header.grid_propagate(False)
        header.columnconfigure(1, weight=1)
        self._reticle(header, 38, INK).grid(row=0, column=0, padx=(28, 14), pady=20)
        identity = tk.Frame(header, bg=INK)
        identity.grid(row=0, column=1, sticky='w')
        tk.Label(identity, text='FOCUS / ANALYSIS', bg=INK, fg=SURFACE,
                 font=('Bahnschrift', 18, 'bold')).pack(anchor='w')
        tk.Label(identity, text='Laser spot measurement workspace', bg=INK,
                 fg='#AFC5CF', font=('Segoe UI', 10)).pack(anchor='w')
        self.open_button = self._button(header, 'Open files', self.select_file, True)
        self.open_button.grid(row=0, column=2, padx=(12, 8), pady=17)
        self.folder_button = self._button(header, 'Open folder', self.select_folder)
        self.folder_button.grid(row=0, column=3, padx=(0, 8), pady=17)
        self.refresh_button = self._button(header, 'Recalculate', self.recalculate)
        self.refresh_button.configure(state='disabled')
        self.refresh_button.grid(row=0, column=4, padx=(0, 28), pady=17)

    def _build_body(self):
        body = tk.Frame(self.root, bg=CANVAS)
        body.grid(row=1, column=0, sticky='nsew', padx=20, pady=(18, 12))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=306)
        body.rowconfigure(1, weight=1)
        heading = tk.Frame(body, bg=CANVAS)
        heading.grid(row=0, column=0, sticky='ew', padx=(4, 16), pady=(0, 8))
        tk.Label(heading, text='Beam profile', bg=CANVAS, fg=INK,
                 font=('Bahnschrift', 19, 'bold')).pack(side='left')
        self.file_label = tk.Label(heading, text='No image loaded', bg=CANVAS,
                                   fg=MUTED, font=('Segoe UI', 10), anchor='e')
        self.file_label.pack(side='right', padx=(12, 0))

        chart_area = tk.Frame(body, bg=CANVAS)
        chart_area.grid(row=1, column=0, sticky='nsew', padx=(0, 12))
        chart_area.columnconfigure(0, weight=1)
        chart_area.rowconfigure(0, weight=1)
        self.plot_scroll = tk.Canvas(chart_area, bg=CANVAS,
                                      highlightthickness=0, borderwidth=0)
        self.plot_scroll.grid(row=0, column=0, sticky='nsew')
        self.plot_scrollbar = ttk.Scrollbar(chart_area, orient='vertical',
                                            style='Plot.Vertical.TScrollbar',
                                            command=self.plot_scroll.yview)
        self.plot_scrollbar.grid(row=0, column=1, sticky='ns')
        self.plot_scrollbar.grid_remove()
        self.plot_scroll.configure(yscrollcommand=self.plot_scrollbar.set)
        self.display_frame = tk.Frame(self.plot_scroll, bg=CANVAS)
        self.plot_window = self.plot_scroll.create_window((0, 0),
                                                           window=self.display_frame, anchor='nw')
        self.plot_scroll.bind('<Configure>', self._resize_plot_area)
        self.display_frame.bind('<Configure>', self._update_scrollregion)
        for column in range(2):
            self.display_frame.columnconfigure(column, weight=1, uniform='plots')
        for row in range(2):
            self.display_frame.rowconfigure(row, weight=1, uniform='plots')
        self._show_empty_state(chart_area)
        self._build_readout(body)

    def _resize_plot_area(self, event):
        self.plot_scroll.itemconfigure(self.plot_window, width=event.width,
                                       height=max(event.height, 700))
        self._update_scrollregion()

    def _update_scrollregion(self, event=None):
        self.plot_scroll.configure(scrollregion=self.plot_scroll.bbox('all'))

    def _scroll_plots(self, event):
        plot = self.plot_scroll
        if (self.current_paths
                and plot.winfo_rootx() <= event.x_root < plot.winfo_rootx() + plot.winfo_width()
                and plot.winfo_rooty() <= event.y_root < plot.winfo_rooty() + plot.winfo_height()):
            plot.yview_scroll(-1 if event.delta > 0 else 1, 'units')

    def _scroll_readout(self, event):
        readout = self.readout_scroll
        if (readout.winfo_rootx() <= event.x_root < readout.winfo_rootx() + readout.winfo_width()
                and readout.winfo_rooty() <= event.y_root < readout.winfo_rooty() + readout.winfo_height()):
            readout.yview_scroll(-1 if event.delta > 0 else 1, 'units')

    def _show_empty_state(self, chart_area):
        self.empty_state = tk.Frame(chart_area, bg=SURFACE,
                                    highlightbackground=RULE, highlightthickness=1)
        self.empty_state.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=6, pady=6)
        center = tk.Frame(self.empty_state, bg=SURFACE)
        center.place(relx=0.5, rely=0.5, anchor='center')
        self._reticle(center, 86, SURFACE).pack(pady=(0, 17))
        tk.Label(center, text='Your beam profile starts here', bg=SURFACE,
                 fg=INK, font=('Bahnschrift', 18, 'bold')).pack()
        tk.Label(center, text='Open one or more CSV/TIFF images, or analyze a folder.',
                 bg=SURFACE, fg=MUTED, font=('Segoe UI', 11)).pack(pady=(7, 20))
        empty_actions = tk.Frame(center, bg=SURFACE)
        empty_actions.pack()
        self.empty_open_button = self._button(empty_actions, 'Open files', self.select_file, True)
        self.empty_open_button.pack(side='left', padx=(0, 8))
        self.empty_folder_button = self._button(empty_actions, 'Open folder', self.select_folder)
        self.empty_folder_button.pack(side='left')

    def _build_readout(self, body):
        rail = tk.Frame(body, bg=SURFACE, width=306,
                        highlightbackground=RULE, highlightthickness=1)
        rail.grid(row=0, column=1, rowspan=2, sticky='nsew')
        rail.grid_propagate(False)
        readout_canvas = tk.Canvas(rail, bg=SURFACE, highlightthickness=0,
                                   borderwidth=0)
        self.readout_scroll = readout_canvas
        readout_canvas.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(rail, orient='vertical',
                                  style='Plot.Vertical.TScrollbar',
                                  command=readout_canvas.yview)
        scrollbar.pack(side='right', fill='y')
        readout_canvas.configure(yscrollcommand=scrollbar.set)
        content = tk.Frame(readout_canvas, bg=SURFACE)
        window = readout_canvas.create_window((0, 0), window=content, anchor='nw')
        content.bind('<Configure>', lambda event: readout_canvas.configure(
            scrollregion=readout_canvas.bbox('all')))
        readout_canvas.bind('<Configure>', lambda event: readout_canvas.itemconfigure(
            window, width=event.width))
        rail = content
        tk.Label(rail, text='BEAM READOUT', bg=SURFACE, fg=MUTED,
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=22, pady=(14, 9))
        self.a0_label = tk.Label(rail, text='a₀', bg=SURFACE, fg=MUTED,
                                 font=('Bahnschrift', 15))
        self.a0_label.pack(anchor='w', padx=22)
        self.a0_value = tk.Label(rail, text='—', bg=SURFACE, fg=ACCENT,
                                 font=('Bahnschrift', 42, 'bold'))
        self.a0_value.pack(anchor='w', padx=22, pady=(0, 8))
        tk.Frame(rail, bg=RULE, height=1).pack(fill='x', padx=22)
        metrics = tk.Frame(rail, bg=SURFACE)
        metrics.pack(fill='x', padx=22, pady=(11, 9))
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        for index, (key, label) in enumerate(self.SINGLE_METRICS):
            cell = tk.Frame(metrics, bg=SURFACE)
            cell.grid(row=index // 2, column=index % 2, sticky='ew',
                      padx=(0, 4) if index % 2 == 0 else (8, 0),
                      pady=(0, 12) if index < 8 else (0, 0))
            metric_label = tk.Label(cell, text=label, bg=SURFACE, fg=MUTED,
                                    font=('Segoe UI', 9))
            metric_label.pack(anchor='w')
            value = tk.Label(cell, text='—', bg=SURFACE, fg=INK,
                             font=('Bahnschrift', 15, 'bold'))
            value.pack(anchor='w', pady=(3, 0))
            self.result_values[key] = value
            self.metric_labels[key] = metric_label
            self.metric_cells[key] = cell
        tk.Frame(rail, bg=RULE, height=1).pack(fill='x', padx=22)
        self.readout_note = tk.Label(
            rail, text='Ellipse overlap: 0–100% (higher means a closer contour match).',
            bg=SURFACE, fg=MUTED, font=('Segoe UI', 9), justify='left',
            wraplength=238)
        self.readout_note.pack(anchor='w', padx=22, pady=(10, 0))

    def _build_parameters(self):
        dock = tk.Frame(self.root, bg=SURFACE,
                        highlightbackground=RULE, highlightthickness=1)
        dock.grid(row=2, column=0, sticky='ew', padx=20, pady=(0, 12))
        tk.Label(dock, text='CALIBRATION & PULSE', bg=SURFACE, fg=INK,
                 font=('Bahnschrift', 12, 'bold')).grid(row=0, column=0, columnspan=5,
                                                        sticky='w', padx=20, pady=(15, 1))
        tk.Label(dock, text='Change a value, then recalculate.', bg=SURFACE,
                 fg=MUTED, font=('Segoe UI', 9)).grid(row=1, column=0, columnspan=5,
                                                      sticky='w', padx=20)
        for column, (key, default) in enumerate(self.DEFAULT_PARAMS.items()):
            dock.columnconfigure(column, weight=1, uniform='params')
            field = tk.Frame(dock, bg=SURFACE)
            field.grid(row=2, column=column, sticky='ew', padx=(20 if column == 0 else 10,
                                                               20 if column == 4 else 10),
                       pady=(15, 19))
            label, unit = self.PARAM_LABELS[key]
            tk.Label(field, text=label, bg=SURFACE, fg=MUTED,
                     font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 6))
            input_row = tk.Frame(field, bg=SURFACE)
            input_row.pack(fill='x')
            entry = tk.Entry(input_row, width=1, font=('Bahnschrift', 13), fg=INK,
                             bg='#F7FAFB', insertbackground=INK, relief='flat',
                             highlightthickness=1, highlightbackground=RULE,
                             highlightcolor=ACCENT, bd=0)
            entry.insert(0, default)
            entry.pack(side='left', fill='x', expand=True, ipady=7)
            tk.Label(input_row, text=unit, bg=SURFACE, fg=MUTED,
                     font=('Segoe UI', 10)).pack(side='left', padx=(7, 0))
            self.param_entries[key] = entry

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=CANVAS)
        footer.grid(row=3, column=0, sticky='ew', padx=24, pady=(0, 10))
        self.status = tk.StringVar(value='Ready · CSV and TIFF supported')
        tk.Label(footer, textvariable=self.status, bg=CANVAS, fg=MUTED,
                 font=('Segoe UI', 9)).pack(side='left')
        tk.Label(footer, text='Right-click a chart to save   ·   Ctrl+O Open files   ·   F5 Recalculate', bg=CANVAS,
                 fg=MUTED, font=('Segoe UI', 9)).pack(side='right')

    def _register_plot(self, panel, figure, source_path, export_name):
        def select(event=None):
            self._select_plot(panel, figure, source_path, export_name)

        def show_menu(event):
            select()
            try:
                self.plot_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.plot_menu.grab_release()

        def bind_children(widget):
            widget.bind('<Button-1>', select, add='+')
            widget.bind('<Button-3>', show_menu, add='+')
            for child in widget.winfo_children():
                bind_children(child)

        bind_children(panel)

    def _select_plot(self, panel, figure, source_path, export_name):
        if self.selected_plot is not None:
            previous_panel = self.selected_plot[0]
            if previous_panel.winfo_exists():
                previous_panel.configure(highlightbackground=RULE)
        panel.configure(highlightbackground=ACCENT)
        self.selected_plot = (panel, figure, source_path, export_name)

    def save_selected_plot(self):
        if self.selected_plot is None:
            return
        panel, figure, source_path, export_name = self.selected_plot
        if not panel.winfo_exists():
            return
        target = filedialog.asksaveasfilename(
            parent=self.root, title='Save chart image',
            initialdir=str(source_path.parent),
            initialfile=f'{source_path.stem}_{export_name}.png',
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('TIFF image', '*.tif'),
                       ('SVG vector image', '*.svg'), ('PDF document', '*.pdf')])
        if not target:
            return
        try:
            figure.savefig(target, dpi=300, bbox_inches='tight', facecolor='white')
        except Exception as exc:
            messagebox.showerror('Save failed', str(exc), parent=self.root)
            self.status.set(f'Save failed · {exc}')
            return
        self.status.set(f'Saved · {Path(target).name}')

    def read_parameters(self):
        values = {}
        for key, entry in self.param_entries.items():
            try:
                value = float(entry.get().strip())
            except ValueError as exc:
                entry.focus_set()
                raise ValueError(f'{self.PARAM_LABELS[key][0]} must be a number.') from exc
            if not np.isfinite(value) or value <= 0:
                entry.focus_set()
                raise ValueError(f'{self.PARAM_LABELS[key][0]} must be greater than zero.')
            values[key] = value
        if values['region_total'] < 1:
            self.param_entries['region_total'].focus_set()
            raise ValueError('Integration radius must be at least 1.')
        return values

    def select_file(self):
        if self.busy:
            return
        selected = filedialog.askopenfilenames(
            title='Open spot images',
            filetypes=[('Spot images', '*.csv *.tif *.tiff'),
                       ('CSV files', '*.csv'), ('TIFF files', '*.tif *.tiff'),
                       ('All files', '*.*')])
        file_paths = tuple(sorted((Path(path) for path in selected),
                                  key=natural_path_key))
        if len(file_paths) == 1:
            self.analyze_file(file_paths[0])
        elif file_paths:
            self.analyze_batch(file_paths)

    def select_folder(self):
        if self.busy:
            return
        folder = filedialog.askdirectory(title='Open folder of spot images')
        if not folder:
            return
        folder_path = Path(folder)
        file_paths = tuple(sorted((
            path for path in folder_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
            key=natural_path_key))
        if not file_paths:
            messagebox.showerror(
                'No spot images',
                'The selected folder contains no CSV, TIF, or TIFF files.',
                parent=self.root)
            return
        selected = self._choose_folder_files(folder_path, file_paths)
        if selected:
            self.analyze_batch(selected)

    def _choose_folder_files(self, folder_path, file_paths):
        dialog = tk.Toplevel(self.root)
        dialog.title('Select spot images')
        dialog.transient(self.root)
        dialog.configure(bg=SURFACE)
        dialog.minsize(520, 420)
        dialog.geometry('620x540')

        tk.Label(dialog, text='SELECT SHOTS', bg=SURFACE, fg=MUTED,
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=22,
                                                     pady=(18, 4))
        tk.Label(dialog, text=folder_path.name, bg=SURFACE, fg=INK,
                 font=('Bahnschrift', 17, 'bold')).pack(anchor='w', padx=22)
        tk.Label(
            dialog,
            text='Choose the files to analyze. Use Ctrl/Shift for multiple selection.',
            bg=SURFACE, fg=MUTED, font=('Segoe UI', 9)).pack(
                anchor='w', padx=22, pady=(5, 12))

        list_frame = tk.Frame(dialog, bg=SURFACE)
        list_frame.pack(fill='both', expand=True, padx=22)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        file_list = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, exportselection=False,
            font=('Consolas', 10), fg=INK, bg='#F7FAFB',
            selectforeground=SURFACE, selectbackground=ACCENT,
            highlightthickness=1, highlightbackground=RULE,
            highlightcolor=ACCENT, relief='flat')
        file_list.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical',
                                  style='Plot.Vertical.TScrollbar',
                                  command=file_list.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        file_list.configure(yscrollcommand=scrollbar.set)
        for path in file_paths:
            file_list.insert('end', path.name)
        file_list.selection_set(0, 'end')

        selected_paths = []

        def confirm():
            indices = file_list.curselection()
            if not indices:
                messagebox.showerror('No files selected',
                                     'Select at least one image to analyze.',
                                     parent=dialog)
                return
            selected_paths.extend(file_paths[index] for index in indices)
            dialog.destroy()

        actions = tk.Frame(dialog, bg=SURFACE)
        actions.pack(fill='x', padx=22, pady=(14, 20))
        tk.Button(actions, text='Select all', command=lambda: file_list.selection_set(0, 'end'),
                  font=('Segoe UI', 10), fg=INK, bg='#E7F0F3', relief='flat',
                  padx=14, pady=8).pack(side='left')
        tk.Button(actions, text='Clear', command=lambda: file_list.selection_clear(0, 'end'),
                  font=('Segoe UI', 10), fg=INK, bg='#E7F0F3', relief='flat',
                  padx=14, pady=8).pack(side='left', padx=(8, 0))
        tk.Button(actions, text='Cancel', command=dialog.destroy,
                  font=('Segoe UI', 10), fg=INK, bg='#E7F0F3', relief='flat',
                  padx=14, pady=8).pack(side='right')
        tk.Button(actions, text='Analyze selected', command=confirm,
                  font=('Segoe UI', 10, 'bold'), fg=SURFACE, bg=ACCENT,
                  activebackground='#0B667F', activeforeground=SURFACE,
                  relief='flat', padx=16, pady=8).pack(side='right', padx=(0, 8))
        dialog.protocol('WM_DELETE_WINDOW', dialog.destroy)
        dialog.grab_set()
        file_list.focus_set()
        self.root.wait_window(dialog)
        return tuple(selected_paths)

    def recalculate(self):
        if self.busy or not self.current_paths:
            return
        if self.current_mode == 'batch':
            self.analyze_batch(self.current_paths)
        else:
            self.analyze_file(self.current_paths[0])

    def _read_and_apply_parameters(self):
        params = self.read_parameters()
        apply_analysis_parameters(params)
        return params

    def _start_worker(self, target, args, message):
        self.busy = True
        self._task_token += 1
        token = self._task_token
        self.open_button.configure(state='disabled')
        self.folder_button.configure(state='disabled')
        self.refresh_button.configure(text='Working…', state='disabled')
        if self.empty_open_button.winfo_exists():
            self.empty_open_button.configure(state='disabled')
        if self.empty_folder_button.winfo_exists():
            self.empty_folder_button.configure(state='disabled')
        self._busy_message = message
        self._animate_busy(token, 0)
        Thread(target=target, args=(*args, token), daemon=True).start()
        self.root.after(40, lambda: self._poll_result(token))

    def analyze_file(self, file_path):
        if self.busy:
            return
        try:
            params = self._read_and_apply_parameters()
        except ValueError as exc:
            messagebox.showerror('Invalid parameter', str(exc), parent=self.root)
            return
        self._start_worker(self._compute_worker, (Path(file_path), params),
                           'Analyzing image')

    def analyze_batch(self, file_paths):
        if self.busy:
            return
        try:
            params = self._read_and_apply_parameters()
        except ValueError as exc:
            messagebox.showerror('Invalid parameter', str(exc), parent=self.root)
            return
        paths = tuple(sorted((Path(path) for path in file_paths),
                             key=natural_path_key))
        self._start_worker(self._compute_batch_worker, (paths, params),
                           f'Analyzing {len(paths)} shots')

    def _animate_busy(self, token, frame):
        if not self.busy or token != self._task_token:
            return
        self.status.set(self._busy_message + '.' * (frame % 3 + 1))
        self.root.after(250, lambda: self._animate_busy(token, frame + 1))

    def _compute_worker(self, file_path, params, token):
        try:
            result = analyze_spot_file(file_path, params, include_profiles=True)
            self._results.put((token, result, None))
        except Exception as exc:
            self._results.put((token, None, exc))

    def _compute_batch_worker(self, file_paths, params, token):
        try:
            results = []
            errors = []
            for shot_number, file_path in enumerate(file_paths, start=1):
                try:
                    shot_result = analyze_spot_file(file_path, params)
                    shot_result['shot_number'] = shot_number
                    results.append(shot_result)
                except Exception as exc:
                    errors.append((shot_number, file_path, str(exc)))
            result = summarize_batch_results(
                results, file_paths, errors, params['PIXEL_WIDTH_MICRONS'])
            self._results.put((token, result, None))
        except Exception as exc:
            self._results.put((token, None, exc))

    def _poll_result(self, token):
        if token != self._task_token or not self.busy:
            return
        try:
            result_token, result, error = self._results.get_nowait()
        except Empty:
            self.root.after(40, lambda: self._poll_result(token))
            return
        if result_token != token:
            self.root.after(40, lambda: self._poll_result(token))
            return
        if error is not None:
            self._finish_analysis(error=error)
            return
        self._busy_message = 'Rendering charts'
        self._render_next(result, 0)

    def _render_next(self, result, index):
        if index == 0:
            self.selected_plot = None
            if self.empty_state.winfo_exists():
                self.empty_state.destroy()
            self.plot_scrollbar.grid()
            for widget in self.display_frame.winfo_children():
                widget.destroy()
        if index == 4:
            self._finish_analysis(result=result)
            return

        try:
            if result['mode'] == 'batch':
                if index == 0:
                    title = 'a₀ · shot series'
                    figure = create_batch_a0_figure(
                        result['shot_numbers'], result['a0_values'])
                elif index == 1:
                    title = 'wₓ / wᵧ · shot series'
                    figure = create_batch_waists_figure(
                        result['shot_numbers'], result['wx_values'],
                        result['wy_values'])
                elif index == 2:
                    title = 'Pointing jitter · Gaussian centers'
                    figure = create_pointing_jitter_figure(
                        result['center_offsets_um'], result['shot_numbers'])
                else:
                    title = 'Average intensity image'
                    figure = create_average_spot_figure(
                        result['mean_data'], result['mean_center_pixel'],
                        result['pixel_width'])
                source_path = result['source_path']
                export_name = self.BATCH_PLOT_EXPORT_NAMES[index]
            else:
                data = result['data']
                x_radius, y_radius, center, orientation, _, major_radius, minor_radius = result['radii']
                region_total = result['region_total']
                pixel_width = result['pixel_width']
                if index == 0:
                    title = 'Ellipse fit · full view'
                    figure = create_ellipse_figure(
                        major_radius, minor_radius, center, orientation, data,
                        region_total, pixel_width, result['fwhm'])
                elif index == 1:
                    title = 'Intensity contours · detail'
                    figure = create_ellipse_zoom_figure(
                        major_radius, minor_radius, center, orientation, data,
                        region_total, pixel_width, result['fwhm'])
                elif index == 2:
                    title = 'a₀ profiles · x / y cuts'
                    figure = create_a0_profile_figure(
                        result['a0_cuts'], x_radius, y_radius, pixel_width)
                else:
                    title = 'a₀ profiles · major / minor cuts'
                    figure = create_a0_axis_profile_figure(
                        result['a0_axis_cuts'], major_radius, minor_radius,
                        pixel_width)
                source_path = result['file_path']
                export_name = self.PLOT_EXPORT_NAMES[index]
            panel = add_plot(self.display_frame, title, figure,
                             index // 2, index % 2)
            self._register_plot(panel, figure, source_path, export_name)
        except Exception as exc:
            self._finish_analysis(error=exc)
            return
        self.root.after(1, lambda: self._render_next(result, index + 1))

    def _show_single_readout(self, result):
        self.a0_label.configure(text='a₀')
        self.a0_value.configure(text=f"{result['a0']:.2f}",
                                font=('Bahnschrift', 42, 'bold'))
        for key, label in self.SINGLE_METRICS:
            self.metric_cells[key].grid()
            self.metric_labels[key].configure(text=label)
        x_radius, y_radius, center, orientation, ellipticity, major_radius, minor_radius = result['radii']
        pixel_width = result['pixel_width']
        self.result_values['x_radius'].configure(text=f'{x_radius * pixel_width:.2f} μm')
        self.result_values['y_radius'].configure(text=f'{y_radius * pixel_width:.2f} μm')
        fwhm = result['fwhm']
        self.result_values['fwhm_x'].configure(text=f"{fwhm['fwhm_x'] * pixel_width:.2f} μm")
        self.result_values['fwhm_y'].configure(text=f"{fwhm['fwhm_y'] * pixel_width:.2f} μm")
        self.result_values['ellipticity'].configure(text=f'{ellipticity:.2f}')
        self.result_values['angle'].configure(text=f"{np.degrees(orientation):.1f}°")
        self.result_values['major_axis'].configure(text=f"{2*fwhm['major']*pixel_width:.2f} μm")
        self.result_values['minor_axis'].configure(text=f"{2*fwhm['minor']*pixel_width:.2f} μm")
        self.result_values['overlap'].configure(text=f"{fwhm['overlap']:.1%}")
        self.result_values['ratio'].configure(text=f"{result['ratio']:.4f}")
        self.readout_note.configure(
            text='Ellipse overlap: 0–100% (higher means a closer contour match).')

    def _show_batch_readout(self, result):
        stats = result['stats']
        self.a0_label.configure(text='Mean a₀ ± 1σ')
        self.a0_value.configure(
            text=f"{stats['a0_mean']:.2f} ± {stats['a0_std']:.2f}",
            font=('Bahnschrift', 27, 'bold'))
        batch_metrics = {
            'x_radius': ('Mean wₓ ± 1σ',
                         f"{stats['wx_mean']:.2f} ± {stats['wx_std']:.2f} μm"),
            'y_radius': ('Mean wᵧ ± 1σ',
                         f"{stats['wy_mean']:.2f} ± {stats['wy_std']:.2f} μm"),
            'fwhm_x': ('X position jitter (1σ)', f"{stats['jitter_x']:.2f} μm"),
            'fwhm_y': ('Y position jitter (1σ)', f"{stats['jitter_y']:.2f} μm"),
            'ellipticity': ('RMS pointing jitter',
                            f"{stats['jitter_rms']:.2f} μm"),
            'angle': ('Analyzed shots', str(len(result['file_paths']))),
            'major_axis': ('Failed files', str(len(result['errors']))),
        }
        for key, _ in self.SINGLE_METRICS:
            if key in batch_metrics:
                label, value = batch_metrics[key]
                self.metric_cells[key].grid()
                self.metric_labels[key].configure(text=label)
                self.result_values[key].configure(text=value)
            else:
                self.metric_cells[key].grid_remove()
        self.readout_note.configure(
            text='Position jitter is the sample standard deviation (1σ) of Gaussian-fit centers.')

    def _finish_analysis(self, result=None, error=None):
        self.busy = False
        self.open_button.configure(state='normal')
        self.folder_button.configure(state='normal')
        if self.empty_open_button.winfo_exists():
            self.empty_open_button.configure(state='normal')
        if self.empty_folder_button.winfo_exists():
            self.empty_folder_button.configure(state='normal')
        self.refresh_button.configure(text='Recalculate',
                                      state='normal' if self.current_paths else 'disabled')
        if error is not None:
            self.status.set(f'Analysis failed · {error}')
            messagebox.showerror('Analysis failed', str(error), parent=self.root)
            return

        if result['mode'] == 'batch':
            self._show_batch_readout(result)
            self.current_mode = 'batch'
            self.current_paths = result['requested_paths']
            self.current_file = result['source_path']
            analyzed = len(result['file_paths'])
            failed = len(result['errors'])
            self.file_label.configure(
                text=f'{analyzed} shots · {result["source_path"].parent.name}')
            status = f'Analyzed {analyzed} shots'
            if failed:
                status += f' · {failed} failed'
            self.status.set(status)
        else:
            self._show_single_readout(result)
            self.current_mode = 'single'
            self.current_paths = (result['file_path'],)
            self.current_file = result['file_path']
            self.file_label.configure(text=self.current_file.name)
            self.status.set(f'Analyzed · {self.current_file.name}')
        self.refresh_button.configure(state='normal')
        self.plot_scroll.yview_moveto(0)


def main():
    root = tk.Tk()
    FocusAnalysisApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
