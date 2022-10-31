import os
import sys
import json

import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

# Import modules from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import config

from processing import calculate_phase_for_fppmeasurement, process_fppmeasurement_with_phasogrammetry, get_phase_field_ROI, get_phase_field_LUT, triangulate_points
from utils import load_fpp_measurements


def fit_to_plane(x, y, z):
    # From https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    tmp_A = []
    tmp_b = []

    for i in range(z.shape[0]):
        tmp_A.append([x[i], y[i], 1])
        tmp_b.append(z[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    fit, residual, rnk, s = linalg.lstsq(A, b)
    return fit


if __name__ == '__main__':

    # Load FPPMeasurements from files
    print('Load FPPMeasurements from files...', end='', flush=True)
    measurements_h = load_fpp_measurements(r'.\data\20-10-2022_13-44-56\measure_20-10-2022_13-44-56.json')
    measurements_v = load_fpp_measurements(r'.\data\20-10-2022_13-45-06\measure_20-10-2022_13-45-06.json')
    print('Done')

    # Load calibration data for cameras stero system
    print('Load calibration data for cameras stereo system...', end='', flush=True)
    with open(config.DATA_PATH + r'/calibrated_data_20-10-2022.json', 'r') as fp:
        calibration_data = json.load(fp)
    print('Done')

    # Calculate phase fields
    print('Calculate phase fields for first camera...', end='', flush=True)
    calculate_phase_for_fppmeasurement(measurements_h[0])
    calculate_phase_for_fppmeasurement(measurements_h[1])
    print('Done')
    print('Calcalute phase fields for second camera...', end='', flush=True)
    calculate_phase_for_fppmeasurement(measurements_v[0])
    calculate_phase_for_fppmeasurement(measurements_v[1])
    print('Done')

    # Plot unwrapped phases
    plt.subplot(221)
    plt.imshow(measurements_h[0].phases[-3], cmap='gray')
    plt.subplot(222)
    plt.imshow(measurements_h[1].phases[-3], cmap='gray')
    plt.subplot(223)
    plt.imshow(measurements_v[0].phases[-3], cmap='gray')
    plt.subplot(224)
    plt.imshow(measurements_v[1].phases[-3], cmap='gray')
    plt.show()

    print('Determine phase fields ROI...', end='', flush=True)
    get_phase_field_ROI(measurements_h[0])
    get_phase_field_ROI(measurements_h[1])
    print('Done')

    # Plot signal to noise ration
    plt.subplot(221)
    plt.imshow(measurements_h[0].modulated_intensities[-1]/measurements_h[0].average_intensities[-1], cmap='gray')
    plt.subplot(222)
    plt.imshow(measurements_h[1].modulated_intensities[-1]/measurements_h[1].average_intensities[-1], cmap='gray')
    plt.subplot(223)
    plt.imshow(measurements_v[0].modulated_intensities[-1]/measurements_v[0].average_intensities[-1], cmap='gray')
    plt.subplot(224)
    plt.imshow(measurements_v[1].modulated_intensities[-1]/measurements_v[1].average_intensities[-1], cmap='gray')
    plt.show()

    print('Calculate phase fields LUT...', end='', flush=True)
    LUT = get_phase_field_LUT(measurements_h[1], measurements_v[1])
    print('Done')

    # Set ROI manually for test plate
    measurements_h[0].ROI = np.array([[430, 270], [1580, 250], [1480, 1300], [500, 1450]], dtype = "float32")

    # Process FPPMeasurements with phasogrammetry approach
    print('Calculate 3D points with phasogrammetry approach...', end='', flush=True)
    points_3d, points_2d_1, points_2d_2, _, _ = process_fppmeasurement_with_phasogrammetry(measurements_h, measurements_v, calibration_data, LUT)
    points_3d = np.reshape(points_3d, (points_3d.shape[0], points_3d.shape[2]))
    print('Done')

    print('Fit points to plane')
    fit = fit_to_plane(points_3d[:,0], points_3d[:,1], points_3d[:,2])
    distance_to_plane = np.abs(points_3d[:, 2] - (fit[0] * points_3d[:, 0] + fit[1] * points_3d[:, 1] + fit[2]))
    
    print(f'Fitting deviation std = {np.std(distance_to_plane)} mm')
    print(f'Fitting deviation mean = {np.mean(distance_to_plane)} mm')

    # plt.hist(distance_to_plane, 30)
    # plt.show()

    # Filter outliers
    print('Try to filter outliers')
    x = points_3d[distance_to_plane<3*np.std(distance_to_plane), 0]
    y = points_3d[distance_to_plane<3*np.std(distance_to_plane), 1]
    z = points_3d[distance_to_plane<3*np.std(distance_to_plane), 2]
    points_2d_1 = points_2d_1[distance_to_plane<3*np.std(distance_to_plane),:]
    points_2d_2 = points_2d_2[distance_to_plane<3*np.std(distance_to_plane),:]

    points_3d, _, _, _, _ = triangulate_points(calibration_data, points_2d_1, points_2d_2)

    print('Fit points to plane without outliers')
    fit2 = fit_to_plane(x, y, z)
    distance_to_plane = np.abs(z - (fit2[0] * x + fit2[1] * y + fit2[2]))
    print(f'Fitting deviation std = {np.std(distance_to_plane)} mm')
    print(f'Fitting deviation mean = {np.mean(distance_to_plane)} mm')

    # plt.hist(distance_to_plane, 30)
    # plt.show()

     # Plot 3D point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    ax.set_xlabel('X, mm')
    ax.set_ylabel('Y, mm')
    ax.set_zlabel('Z, mm')

    ax.set_ylim(ax.get_xlim())

    ax.view_init(elev=-75, azim=-89)

    plt.show()

    plt.tricontourf(x, y, distance_to_plane, levels=100)
    plt.colorbar()
    plt.show()
