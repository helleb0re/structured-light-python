import os
import sys
import json

import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

# Import modules from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import config
from fpp_structures import PhaseShiftingAlgorithm
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
    measurements_h = load_fpp_measurements(r'.\data\01-11-2022_17-55-30\fpp_measurement.json')
    measurements_v = load_fpp_measurements(r'.\data\01-11-2022_17-55-40\fpp_measurement.json')
    print('Done')

    # Display FPPMeasurement parameters
    if measurements_h[0].phase_shifting_type == PhaseShiftingAlgorithm.n_step:
        algortihm_type = f'{len(measurements_h[0].shifts)}-step' 
    elif measurements_h[0].phase_shifting_type == PhaseShiftingAlgorithm.double_three_step:
        algortihm_type = 'double 3-step'
    print(f'\nPhase shift algorithm: {algortihm_type}')
    print(f'Phase shifts: {measurements_h[0].shifts}')
    print(f'Frequencies: {measurements_h[0].frequencies}\n')

    # Load calibration data for cameras stero system
    print('Load calibration data for cameras stereo system...', end='', flush=True)
    with open(config.DATA_PATH + r'/calibrated_data_01-11-2022.json', 'r') as fp:
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
    plt.imshow(measurements_h[0].unwrapped_phases[-1], cmap='gray')
    plt.subplot(222)
    plt.imshow(measurements_h[1].unwrapped_phases[-1], cmap='gray')
    plt.subplot(223)
    plt.imshow(measurements_v[0].unwrapped_phases[-1], cmap='gray')
    plt.subplot(224)
    plt.imshow(measurements_v[1].unwrapped_phases[-1], cmap='gray')
    plt.show()

    print('Determine phase fields ROI...', end='', flush=True)
    get_phase_field_ROI(measurements_h[0])
    get_phase_field_ROI(measurements_h[1])
    print('Done')

    # Set ROI manually for test plate
    measurements_h[0].ROI = np.array([[470, 230], [1420, 170], [1350, 1150], [520, 1350]], dtype = "float32")

    # Plot signal to noise ration
    plt.subplot(221)
    plt.imshow(measurements_h[0].modulated_intensities[-1]/measurements_h[0].average_intensities[-1], cmap='gray')
    # Draw ROI
    plt.plot(measurements_h[0].ROI[:, 0], measurements_h[0].ROI[:, 1], 'r-')
    plt.plot([measurements_h[0].ROI[-1, 0], measurements_h[0].ROI[0, 0]],
             [measurements_h[0].ROI[-1, 1], measurements_h[0].ROI[0, 1]], 'r-')
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

    # Process FPPMeasurements with phasogrammetry approach
    print('Calculate 2D corresponding points with phasogrammetry approach...')
    points_2d_1, points_2d_2 = process_fppmeasurement_with_phasogrammetry(measurements_h, measurements_v, 5, 5, LUT)
    print(f'Found {points_2d_1.shape[0]} corresponding points')
    print('Done')

    print('Calculate 3D points with triangulation...')
    points_3d, rms1, rms2, reproj_err1, reproj_err2 = triangulate_points(calibration_data, points_2d_1, points_2d_2)
    print(f'Reprojected RMS for camera 1 = {rms1:.3f}')
    print(f'Reprojected RMS for camera 2 = {rms2:.3f}')
    print('Done')

    print('\nFit points to plane')
    fit = fit_to_plane(points_3d[:,0], points_3d[:,1], points_3d[:,2])
    distance_to_plane = np.abs(points_3d[:, 2] - (fit[0] * points_3d[:, 0] + fit[1] * points_3d[:, 1] + fit[2]))
    
    print(f'Fitting deviation mean = {np.mean(distance_to_plane):.4f} mm')
    print(f'Fitting deviation max = {np.max(distance_to_plane):.4f} mm')
    print(f'Fitting deviation std = {np.std(distance_to_plane):.4f} mm')

    # plt.hist(distance_to_plane, 30)
    # plt.show()

    # Filter outliers by reprojection error
    reproj_err_threshold = 1.0 # pixel

    print('\nTry to filter outliers with reprojection error threshold...')
    filter_condition = (reproj_err1 < reproj_err_threshold) & (reproj_err2 < reproj_err_threshold)
    x = points_3d[filter_condition, 0]
    y = points_3d[filter_condition, 1]
    z = points_3d[filter_condition, 2]
    points_2d_1 = points_2d_1[filter_condition,:]
    points_2d_2 = points_2d_2[filter_condition,:]
    print(f'Found {points_3d.shape[0] - x.shape[0]} outliers')

    print('Calculate 3D points with triangulation without outliers...')
    points_3d, rms1, rms2, reproj_err1, reproj_err2 = triangulate_points(calibration_data, points_2d_1, points_2d_2)
    print(f'Reprojected RMS for camera 1 = {rms1:.3f}')
    print(f'Reprojected RMS for camera 2 = {rms2:.3f}')
    print('Done')

    print('\nFit points to plane without outliers')
    fit2 = fit_to_plane(x, y, z)
    distance_to_plane = np.abs(z - (fit2[0] * x + fit2[1] * y + fit2[2]))
    print(f'Fitting deviation mean = {np.mean(distance_to_plane):.4f} mm')
    print(f'Fitting deviation max = {np.max(distance_to_plane):.4f} mm')
    print(f'Fitting deviation std = {np.std(distance_to_plane):.4f} mm\n')

    print('\nTry to filter outliers with distance to fitted surface...')
    filter_condition = distance_to_plane < 3*np.std(distance_to_plane)
    x = points_3d[filter_condition, 0]
    y = points_3d[filter_condition, 1]
    z = points_3d[filter_condition, 2]
    points_2d_1 = points_2d_1[filter_condition,:]
    points_2d_2 = points_2d_2[filter_condition,:]
    print(f'Found {points_3d.shape[0] - x.shape[0]} outliers')

    print('\nFit points to plane without outliers')
    fit2 = fit_to_plane(x, y, z)
    distance_to_plane = np.abs(z - (fit2[0] * x + fit2[1] * y + fit2[2]))
    print(f'Fitting deviation mean = {np.mean(distance_to_plane):.4f} mm')
    print(f'Fitting deviation max = {np.max(distance_to_plane):.4f} mm')
    print(f'Fitting deviation std = {np.std(distance_to_plane):.4f} mm\n')

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
