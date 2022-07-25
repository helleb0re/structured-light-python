'''Module to process FPP images'''

from __future__ import annotations
import multiprocessing
from typing import Optional
from multiprocessing import Pool

import cv2
import numpy as np
from scipy import signal

import config
from fpp_structures import FPPMeasurement 


def calculate_phase_generic(images: list[np.ndarray], phase_shifts: Optional[list[float]]=None, frequency: Optional[float]=None, direct_formula: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] :
    '''
    Calculate wrapped phase from several PSP images by 
    generic formula (8) in https://doi.org/10.1016/j.optlaseng.2018.04.019

    Args:
        images (list): the list of PSP images
        phase_shifts=None (list): the list of phase shifts for each image from images,
        if phase_shifts is not defined, its calculated automatical for uniform step
        frequency=None (float): the frequency of measurement to add PI for unity frequency images
        direct_formula=False (bool): use direct formulas to calculate phases for 3 and 4 phase shifts

    Returns:
        result_phase (2D numpy array): wrapped phase from images
        average_intensity (2D numpy array): average intensity on images
        modulated_intensity (2D numpy array): modulated intensity on images
    '''

    assert phase_shifts is None or len(images) == len(phase_shifts), \
    'Length of phase_shifts must be equal to images length'

    # Calculate shifts if its not defined 
    if phase_shifts is None:
        phase_shifts = [2 * np.pi / len(images) * n for n in range(len(images))]

    # Form arrays for broadcasting
    imgs = np.zeros((len(images), images[0].shape[0], images[0].shape[1]))

    for i in range(len(images)):
        imgs[i] = images[i]

    # Use specific case for phase shifts length
    if direct_formula and len(phase_shifts) == 3:
        # Calculate formula (14-16) in https://doi.org/10.1016/j.optlaseng.2018.04.019
        sum12 = imgs[1] - imgs[2]
        sum012 = 2 * imgs[0] - imgs[1] - imgs[2]
        result_phase = np.arctan2(np.sqrt(3) * (sum12), sum012)
        average_intensity = (imgs[0] + imgs[1] + imgs[2]) / 3
        modulated_intensity = 1/3 * np.sqrt(3*(sum12)**2 + (sum012)**2)
    elif direct_formula and len(phase_shifts) == 4:
        # Calculate formula (21-23) in https://doi.org/10.1016/j.optlaseng.2018.04.019
        sum13 = imgs[1] - imgs[3]
        sum02 = imgs[0] - imgs[2]
        result_phase = np.arctan2(sum13, sum02)
        average_intensity = (imgs[0] + imgs[1] + imgs[2] + imgs[3]) / 4
        modulated_intensity = 0.5 * np.sqrt(sum13**2 + sum02**2)
    else:
        # Reshape phase shifts for broadcasting multiplying
        phase_shifts = np.array(phase_shifts).reshape((-1,) + (1, 1))

        # Add suplementary phase to get phase for unity frequency measurment
        phase_sup = 0
        if frequency is not None and frequency == 1:
            phase_sup = np.pi

        # Calculate formula (8) in https://doi.org/10.1016/j.optlaseng.2018.04.019
        temp1 = np.multiply(imgs, np.sin(phase_shifts + phase_sup))
        temp2 = np.multiply(imgs, np.cos(phase_shifts + phase_sup))

        sum1 = np.sum(temp1, 0)
        sum2 = np.sum(temp2, 0)

        result_phase = np.arctan2(sum1, sum2)

        # Calculate formula (9-10) in https://doi.org/10.1016/j.optlaseng.2018.04.019
        average_intensity = np.mean(imgs, 0)
        modulated_intensity = 2 * np.sqrt(np.power(sum1, 2) + np.power(sum2, 2)) / len(images)

    return result_phase, average_intensity, modulated_intensity


def calculate_unwraped_phase(phase_l: np.ndarray, phase_h: np.ndarray, lamb_l:float , lamb_h: float) -> np.ndarray:
    '''
    Calculate unwrapped phase from two sets of PSP images by 
    formula (94-95) in https://doi.org/10.1016/j.optlaseng.2018.04.019
    with standard temporal phase unwrapping (TPU) algorithm

    Args:
        phase_l (2D numpy array): The calculated phase for set of PSP images with low frequency (lamb_l) 
        phase_h (2D numpy array): The calculated phase for set of PSP images with high frequency (lamb_h) 
        lamb_l (float): The low spatial frequency for first phase array (phase_l)
        lamb_h (float): The high spatial frequency for second phase array (phase_h)

    Returns:
        unwrapped_phase (2D numpy array): unwrapped phase
    '''
    assert phase_h.shape == phase_l.shape, \
    'Shapes of phase_l and phase_h must be equals'

    # Formula (95) in https://doi.org/10.1016/j.optlaseng.2018.04.019
    k = np.round(((lamb_l / lamb_h) * phase_l - phase_h) / (2 * np.pi)).astype(int)

    # Formula (94) in https://doi.org/10.1016/j.optlaseng.2018.04.019
    unwrapped_phase = phase_h + 2 * np.pi * k

    return unwrapped_phase


def load_image(path: str) -> np.ndarray:
    '''
    Load image from file
    
    Args:
        path (string): path to file for loading

    Returns:
        image (2D numpy array): loaded image
    '''
    image = cv2.imread(path)

    # Tranform image to grayscale
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def calculate_phase_for_fppmeasurement(measurement: FPPMeasurement) -> tuple[list[np.ndarray], list[np.ndarray]]:
    '''
    Calculate unwrapped phase for FPP measurement instance with the help
    of calculate_phase_generic and calculate_unwraped_phase functions    

    Args:
        measurement (FPPMeasurement): FPP measurement instance

    Returns:
        phases (List of 2D numpy array): wrapped phases
        unwrapped_phases (List of 2D numpy array): unwrapped phase
    '''
    # Load measurement data
    shifts_count = measurement.shifts_count
    frequencies = measurement.frequencies
    frequency_counts = measurement.frequency_counts
    images = measurement.imgs_list

    phases = []
    unwrapped_phases = []
    avg_ints = []
    mod_ints = []

    for i in range(frequency_counts):
        images_for_one_frequency = []

        if images is None:
            for j in range(shifts_count):
                im = load_image(measurement.imgs_file_names[i][j])
                images_for_one_frequency.append(im)
        else:
            images_for_one_frequency = images[i]

        phase, avg_int, mod_int = calculate_phase_generic(images_for_one_frequency, measurement.shifts, measurement.frequencies[i])

        mask = np.where(mod_int > 5, 1, 0) 
        phase = phase * mask

        phases.append(phase)
        avg_ints.append(avg_int)
        mod_ints.append(mod_int)
        
        if i == 0:
            unwrapped_phases.append(phase)
        else:
            unwrapped_phase = calculate_unwraped_phase(unwrapped_phases[i-1], phases[i], 1 / frequencies[i-1], 1 / frequencies[i])
            unwrapped_phases.append(unwrapped_phase)
    
    return phases, unwrapped_phases, avg_ints, mod_ints


def point_inside_polygon(x: int, y: int, poly: list[tuple(int, int)] , include_edges: bool = True) -> bool:
    '''
    Test if point (x,y) is inside polygon poly

    Point is inside polygon if horisontal beam to the right
    from point crosses polygon even number of times. Works fine for non-convex polygons.

    Args:
        x (int): horizontal point coordinate
        y (int): vertical point coordinate
        poly (list[tuple(int, int)]): N-vertices polygon defined as [(x1,y1),...,(xN,yN)] or [(x1,y1),...,(xN,yN),(x1,y1)]

    Returns:
        inside (bool): if point inside the polygon
    '''
    n = len(poly)

    inside = False

    p1x, p1y = poly[0]

    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if p1y == p2y:
            if y == p1y:
                if min(p1x, p2x) <= x <= max(p1x, p2x):
                    # point is on horisontal edge
                    inside = include_edges
                    break
                # point is to the left from current edge
                elif x < min(p1x, p2x):  
                    inside = not inside
        else:  # p1y!= p2y
            if min(p1y, p2y) <= y <= max(p1y, p2y):
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                # point is right on the edge
                if x == xinters:  
                    inside = include_edges
                    break
                
                # point is to the left from current edge
                if x < xinters:  
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside


def find_phasogrammetry_corresponding_point(p1_h: np.ndarray, p1_v: np.ndarray, p2_h: np.ndarray, p2_v: np.ndarray, x: int, y: int) -> tuple[float, float]:
    '''
    Finds the corresponding point coordinates for the second image using the phasogrammetry approach 

    For the given coordinates x and y, the phase values on the fields for the vertical and horizontal fringes 
    for the images of the first camera are determined. Then two isolines with defined values of the phase on 
    the corresponding fields for the second camera are found. The intersection of the isolines gives the 
    coordinates of the corresponding point on the image from the second camera.

    Args:
        p1_h (numpy array): phase field for horizontal fringes for first camera
        p1_v (numpy array): phase field for vertical fringes for first camera
        p2_h (numpy array): phase field for horizontal fringes for second camera
        p2_v (numpy array): phase field for vertical fringes for second camera
        x (int): horizontal coordinate of point for first camera
        y (int): vertical coordinate of point for first camera

    Returns:
        x2, y2 (tuple[float, float]): horizontal and vertical coordinate of corresponding point for second camera
    '''
    # Determine the phase values on vertical and horizontal phase fields 
    phase_h = p1_h[y, x]
    phase_v = p1_v[y, x]

    # Find coords of isophase curves
    y_h, x_h = np.where(np.isclose(p2_h, phase_h, atol=10**-1))
    y_v, x_v = np.where(np.isclose(p2_v, phase_v, atol=10**-1))

    # A faster way to calculate using a flatten array 
    # _, yx_h = np.unravel_index(np.where(np.isclose(p2_h, p1_h[y, x], atol=10**-1)), p2_h.shape)
    # _, yx_v = np.unravel_index(np.where(np.isclose(p2_v, p1_v[y, x], atol=10**-1)), p2_v.shape)

    # Find ROI of coords for intersection
    y_h_min = np.min(y_h)
    y_h_max = np.max(y_h)
    x_v_min = np.min(x_v)
    x_v_max = np.max(x_v)

    # Apply ROI for coords of isophase curves
    y_h = y_h[(x_h >= x_v_min) & (x_h <= x_v_max)]
    x_h = x_h[(x_h >= x_v_min) & (x_h <= x_v_max)]
    x_v = x_v[(y_v >= y_h_min) & (y_v <= y_h_max)]
    y_v = y_v[(y_v >= y_h_min) & (y_v <= y_h_max)]

    # Break if too much points in isophase line
    if len(y_h) > 500 or len(y_v) > 500:
        return -1, -1

    # Break if no points found
    if x_h.size == 0 or x_v.size == 0:
        return -1, -1

    # Reshape coords to use broadcasting
    x_h = x_h[:, np.newaxis]
    y_h = y_h[:, np.newaxis]
    y_v = y_v[np.newaxis, :]
    x_v = x_v[np.newaxis, :]

    # Calculate distance between points in coords
    distance = np.sqrt((x_h - x_v)**2 + (y_h - y_v)**2)

    # Find indicies of minimum distance
    i_h_min, i_v_min = np.where(distance == distance.min())
    i_v_min = i_v_min[0]
    i_h_min = i_h_min[0]

    # A faster way to calculate using a flatten array 
    # i_h_min, i_v_min = np.unravel_index(np.where(distance.ravel()==distance.min()), distance.shape)
    # i_v_min = i_v_min[0][0]
    # i_h_min = i_h_min[0][0]

    x2, y2 = ((x_v[0, i_v_min] + x_h[i_h_min, 0]) / 2, (y_v[0, i_v_min] + y_h[i_h_min, 0]) / 2)
    return x2, y2


def process_fppmeasurement_with_phasogrammetry(measurements_h: list[FPPMeasurement], measurements_v: list[FPPMeasurement], calibration_data: dict) -> tuple[np.ndarray]:
    '''
    Find 3D point cloud with phasogrammetry approach 

    Args:
        fppmeasurements_h (list of FPPMeasurement): list of FPPMeasurements instances with horizontal fringes
        fppmeasurements_v (list of FPPMeasurement): list of FPPMeasurements instances with vertical fringes
        calibration_data (dict): dict of calibration data for stereo cameras system
    Returns:
        points_3d (numpy aaray): 3D point cloud
        rms1 (float): reprojection error for first camera
        rms2 (float): reprojection error for second camera
    '''
    # TODO: Remove the phase calculation code
    # Calculate phase fields
    print('Calculate phase fields for first camera...', end='')
    _, unwrapped_phases1_h, _, _ = calculate_phase_for_fppmeasurement(measurements_h[0])
    _, unwrapped_phases2_h, _, _ = calculate_phase_for_fppmeasurement(measurements_h[1])
    print('Done')
    print('Calcalute phase fields for second camera...', end='')
    _, unwrapped_phases1_v, _, _ = calculate_phase_for_fppmeasurement(measurements_v[0])
    _, unwrapped_phases2_v, _, _ = calculate_phase_for_fppmeasurement(measurements_v[1])
    print('Done')

    # Take phases with highest frequencies 
    p1_h = unwrapped_phases1_h[-1]
    p2_h = unwrapped_phases2_h[-1]

    p1_v = unwrapped_phases1_v[-1]
    p2_v = unwrapped_phases2_v[-1]

    # TODO: Automatic ROI detection
    # Сoordinates of the corners of the rectangle to set the ROI processing
    srcTri = np.array([[205, 427], [1754, 238], [1777, 1297], [132, 1308]], dtype = "float32")
    dstTri = np.array([[116, 159], [1890, 284], [1967, 1258], [53, 1243]], dtype = "float32")

    # Cut ROI from phase fields for second camera
    ROIx = slice(int(np.min(dstTri[:,0])), int(np.max(dstTri[:,0])))
    ROIy = slice(int(np.min(dstTri[:,1])), int(np.max(dstTri[:,1])))

    # p1_h = p1_h[ROIy][ROIx]
    # p1_v = p1_v[ROIy][ROIx]
    p2_h = p2_h[ROIy, ROIx]
    p2_v = p2_v[ROIy, ROIx]

    # Calculation of the coordinate grid on first image
    xx = np.arange(0, p1_h.shape[1], 50, dtype=np.int32)
    yy = np.arange(0, p1_h.shape[0], 50, dtype=np.int32)

    coords1 = []

    for y in yy:
        for x in xx:
            # Check if coordinate in ROI rectangle
            if point_inside_polygon(x, y, srcTri):
                coords1.append((x, y))

    coords2 = []

    print(f'Start calculating phase correlation for {len(coords1)} points')

    coords_to_delete = []

    if config.USE_MULTIPROCESSING:
        # Use parallel calaculation to increase processing speed 
        with multiprocessing.Pool(6) as p:
            coords2 = p.starmap(find_phasogrammetry_corresponding_point, [(p1_h, p1_v, p2_h, p2_v, coords1[i][0], coords1[i][1]) for i in range(len(coords1))])

        # Search for corresponding points not found
        for i in range(len(coords2)):
            if coords2[i][0] < 0 and coords2[i][1] < 0:
                coords_to_delete.append(i)
            else:
                # Add ROI left and top coordinates
                coords2[i] = (coords2[i][0] + ROIx.start, coords2[i][1] + ROIy.start)

        # If no point found, delete coordinate from grid
        for index in reversed(coords_to_delete):
            coords1.pop(index)
            coords2.pop(index)
    else:
        for i in range(len(coords1)):
            # Find corresponding point coordinate on second image
            x, y = find_phasogrammetry_corresponding_point(p1_h, p1_v, p2_h, p2_v, coords1[i][0], coords1[i][1])
            # If no point found, delete coordinate from grid
            if x == -1 and y == -1:
                coords_to_delete.append(i)
            else:
                coords2.append((x + ROIx.start, y + ROIy.start))
            print(f'Found {i+1} from {len(coords1)} points')

        # Delete point in grid with no coresponding point on second image
        for index in reversed(coords_to_delete):
            coords1.pop(index)

    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    # Form a set of coordinates of corresponding points on the first and second images
    image1_points = []
    image2_points = []

    for point1, point2 in zip(coords1, coords2):
        image1_points.append([point1[0], point1[1]]) 
        image2_points.append([point2[0], point2[1]])
        
    print(f'Start triangulating points...')

    # TODO: Extract triangulation into a separate method
    # Calculate the projective matrices according to the stereo calibration data
    proj_mtx_1 = np.dot(calibration_data['camera_0']['mtx'], np.hstack((np.identity(3), np.zeros((3,1)))))
    proj_mtx_2 = np.dot(calibration_data['camera_1']['mtx'], np.hstack((calibration_data['R'], calibration_data['T'])))

    # Calculate the triangulation of 3D points
    image_points_nparray = np.array(image1_points, dtype=float).T
    image_points_nparray2 = np.array(image2_points, dtype=float).T
    points_hom = cv2.triangulatePoints(proj_mtx_1, proj_mtx_2, image_points_nparray, image_points_nparray2)
    points_3d = cv2.convertPointsFromHomogeneous(points_hom.T)

    # Reproject triangulated points
    reproj_points, _ = cv2.projectPoints(points_3d, np.identity(3), np.zeros((3,1)),
                                                np.array(calibration_data['camera_0']['mtx']), np.array(calibration_data['camera_0']['dist']))

    reproj_points2, _ = cv2.projectPoints(points_3d, np.array(calibration_data['R']), np.array(calibration_data['T']),
                                                np.array(calibration_data['camera_1']['mtx']), np.array(calibration_data['camera_1']['dist']))

    # Calculate reprojection error
    tot_error = 0
    tot_error += np.sum(np.square(np.float64(image_points_nparray - reproj_points[:,0,:].T)))
    rms1 = np.sqrt(tot_error/len(reproj_points))
    print(f'Reprojected RMS for camera 1 = {rms1:.3f}')
    tot_error = 0
    tot_error += np.sum(np.square(np.float64(image_points_nparray2 - reproj_points2[:,0,:].T)))
    rms2 = np.sqrt(tot_error/len(reproj_points))
    print(f'Reprojected RMS for camera 2 = {rms2:.3f}')
 
    return points_3d, rms1, rms2


def calculate_displacement_field(field1: np.ndarray, field2: np.ndarray, win_size_x: int, win_size_y: int, step_x: int, step_y: int) -> np. ndarray:
    '''
    Calculate displacement field between two scalar fields thru correlation.

    Args:
        field1 (2D numpy array): first scalar field
        field2 (2D numpy array): second scalar field
        win_size_x (int): interrogation window horizontal size
        win_size_y (int): interrogation window vertical size
        step_x (int): horizontal step for dividing on interrogation windows
        step_y (int): vertical step for dividing on interrogation windows
    Returns:
        vector_field (): vector field of displacements
    '''
    assert field1.shape == field2.shape, 'Shapes of field1 and field2 must be equals'
    assert win_size_x > 4 and win_size_y > 4, 'Size of interrogation windows should be greater than 4 pixels'
    assert step_x > 0 and step_y > 0, 'Horizontal and vertical steps should be greater than zero'

    # Get interrogation windows
    list_of_windows = [[], []]
    list_of_coords = []
    
    width = field1.shape[1]
    height = field1.shape[0]
    num_win_x = range(int(np.floor((width - win_size_x)/step_x + 1)))
    num_win_y = range(int(np.floor((height - win_size_y)/step_y + 1)))
       
    for i in num_win_x:
        start_x = step_x * i
        end_x = step_x * i + win_size_x
        center_x = np.round(end_x - win_size_x / 2)

        for j in num_win_y:
            start_y = step_y * j
            end_y = step_y * j + win_size_y
            center_y = np.round(end_y - win_size_y / 2)

            window1 = field1[start_y:end_y, start_x:end_x]
            window2 = field2[start_y:end_y, start_x:end_x]
            list_of_windows[0].append(window1)
            list_of_windows[1].append(window2)
            list_of_coords.append([center_x, center_y])

    # Calculate correlation function
    correlation_list = []

    # Create 2D Gauss kernel
    gauss = np.outer(signal.windows.gaussian(win_size_x, win_size_x / 2),
                     signal.windows.gaussian(win_size_y, win_size_y / 2))

    for i in range(len(list_of_windows[0])):
        # Windowing interrogation windows
        list_of_windows[0][i] = list_of_windows[0][i] * gauss
        list_of_windows[1][i] = list_of_windows[1][i] * gauss
        mean1 = np.mean(list_of_windows[0][i])
        std1 = np.std(list_of_windows[0][i])
        mean2 = np.mean(list_of_windows[1][i])
        std2 = np.std(list_of_windows[1][i])
        a = np.fft.rfft2(list_of_windows[0][i] - mean1, norm='ortho')
        b = np.fft.rfft2(list_of_windows[1][i] - mean2, norm='ortho')
        c = np.multiply(a, b.conjugate())
        d = np.fft.irfft2(c)
        if std1 == 0:
            std1 = 1
        if std2 == 0:
            std2 = 1
        e = d / (std1 * std2)
        correlation_list.append(e)

    # Find maximums
    maximums_list = []

    for i in range(len(correlation_list)):
        
        # Find maximum indexes for x and y
        maximum = np.unravel_index(correlation_list[i].argmax(), correlation_list[i].shape)                        

        # Get neighborhood pixels of maximum at X axis 
        cx0 = np.fabs(correlation_list[i][maximum[0], maximum[1] - 1])
        cx1 = np.fabs(correlation_list[i][maximum[0], maximum[1]    ])

        if (maximum[1] == correlation_list[i].shape[1]):
            cx2 = np.fabs(correlation_list[i][maximum[0], maximum[1] + 1])
        else:
            cx2 = np.fabs(correlation_list[i][maximum[0], 0])

        # Get neighborhood pixels of maximum at Y axis 
        cy0 = np.fabs(correlation_list[i][maximum[0] - 1, maximum[1]])
        cy1 = np.fabs(correlation_list[i][maximum[0]    , maximum[1]])

        if (maximum[0] == correlation_list[i].shape[0]):
            cy2 = np.fabs(correlation_list[i][maximum[0] + 1, maximum[1]])
        else:
            cy2 = np.fabs(correlation_list[i][0, maximum[1]])

        # 3-point gauss fit
        try:
            x_max = maximum[1] + (np.log(np.abs(cx0))  - np.log(np.abs(cx2)))/(2 * np.log(np.abs(cx0)) - 4 * np.log(np.abs(cx1)) + 2 * np.log(np.abs(cx2)))
        except (ZeroDivisionError, ValueError):
            x_max = 0
        try:
            y_max = maximum[0] + (np.log(np.abs(cy0))  - np.log(np.abs(cy2)))/(2 * np.log(np.abs(cy0)) - 4 * np.log(np.abs(cy1)) + 2 * np.log(np.abs(cy2)))
        except (ZeroDivisionError, ValueError):
            y_max = 0


        # Shift maximum due to pereodic of correlation function
        if x_max > correlation_list[i].shape[0] / 2:
            x_max = x_max - correlation_list[i].shape[0]
        elif np.fabs(x_max) < 0.01:
            x_max = 0

        # Shift maximum due to pereodic of correlation function
        if y_max > correlation_list[i].shape[1] / 2:
            y_max = y_max - correlation_list[i].shape[1]
        elif np.fabs(y_max) < 0.01:
            y_max = 0

        # Not actual maximum value
        maximums_list.append([x_max, y_max, np.max(correlation_list[i])])

    # Create vector field
    vector_field = []

    return np.array(list_of_coords), np.array(maximums_list)