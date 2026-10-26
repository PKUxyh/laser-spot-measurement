import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from scipy.signal import convolve2d
from skimage.transform import hough_ellipse
from skimage.restoration import estimate_sigma
from zernike_rzern import RZern  # updated import
from data_io import load_spot_data




def preprocess_data(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data[data < 0] = 0
    # data = gaussian_filter(data, sigma=7)      #高斯滤波不合适，会让计算偏差很大
    return data

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
    radius = np.sqrt(region.area / np.pi)

    return radius, (x0, y0)

def zernike_analysis(data):
    radius, center = calculate_radius(data)
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2) / radius
    theta = np.arctan2(y - center[1], x - center[0])

    zernike = RZern(10)
    zernike.make_cart_grid(r, theta)
    coeffs = zernike.fit(data)

    return coeffs, zernike

def display_zernike_coeffs(coeffs):
    fig, ax = plt.subplots()
    ax.bar(range(len(coeffs)), np.abs(coeffs))
    ax.set_yscale('log')
    ax.set_xlabel('Zernike Polynomial Index')
    ax.set_ylabel('Coefficient Magnitude')
    ax.set_title('Zernike Coefficients')
    plt.show()


def main(file_path):
    data = load_spot_data(file_path)
    data = preprocess_data(data)

    plt.imshow(data, cmap='viridis', norm=LogNorm())
    plt.colorbar()
    plt.title('Focal Spot Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

    radius, center = calculate_radius(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.sqrt((i - center[1])**2 + (j - center[0])**2) > 2*radius:
                data[i, j] = 0

    x_min = max(int(center[0] - 2*radius), 0)
    x_max = min(int(center[0] + 2*radius), data.shape[1])
    y_min = max(int(center[1] - 2*radius), 0)
    y_max = min(int(center[1] + 2*radius), data.shape[0])

    data = data[y_min:y_max, x_min:x_max]

    plt.imshow(data, cmap='viridis', norm=LogNorm())
    plt.colorbar()
    plt.title('Processed Focal Spot Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

    coeffs, zernike = zernike_analysis(data)
    display_zernike_coeffs(coeffs)
    
    reconstructed_image = zernike.eval_grid(coeffs, matrix=True)
    
    plt.imshow(reconstructed_image, cmap='viridis', norm=LogNorm())
    plt.colorbar()
    plt.title('Reconstructed Focal Spot Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit a CSV or grayscale TIFF spot image with Zernike polynomials.')
    parser.add_argument('file', help='Path to a CSV, TIF, or TIFF file')
    main(parser.parse_args().file)
