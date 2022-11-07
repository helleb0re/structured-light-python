'''Module to process FPP images'''

from __future__ import annotations
import multiprocessing
from typing import Optional
from multiprocessing import Pool

import cv2
import numpy as np
from scipy import signal
from scipy.optimize import fsolve

import config
from fpp_structures import FPPMeasurement, PhaseShiftingAlgorithm 


def calculate_phase_generic(images: list[np.ndarray], phase_shifts: Optional[list[float]]=None, frequency: Optional[float]=None, phase_shifting_type: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.n_step, direct_formula: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] :
    '''
    Calculate wrapped phase from several Phase Shifting Profilometry images by 
    generic formula (8) in https://doi.org/10.1016/j.optlaseng.2018.04.019

    Args:
        images (list of numpy arrays): the list of Phase Shifting Profilometry images
        phase_shifts = None (list): the list of phase shifts for each image from images,
        if phase_shifts is not defined, its calculated automatical for uniform step
        frequency = None (float): the frequency of measurement to add PI for unity frequency images
        phase_shifting_type = n_step (enum(int)): type of phase shifting algorithm should be used for phase calculating
        direct_formula = False (bool): use direct formulas to calculate phases for 3- and 4-step phase shifts

    Returns:
        result_phase (2D numpy array): wrapped phase from images
        average_intensity (2D numpy array): average intensity on images
        modulated_intensity (2D numpy array): modulated intensity on images
    '''
    def calculate_n_step_phase(imgs: list[np.ndarray], phase_shifts: list[float]):
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
    
    # Calculate shifts if its not defined 
    if phase_shifts is None:
        phase_shifts = [2 * np.pi / len(images) * n for n in range(len(images))]

    # Form numpy array for broadcasting
    imgs = np.zeros((len(images), images[0].shape[0], images[0].shape[1]))

    # Add images to formed numpy array
    for i in range(len(images)):
        imgs[i] = images[i]
    
    # Depending on phase shift algorithm calculate wrapped phase field
    if phase_shifting_type == PhaseShiftingAlgorithm.n_step:
        # Classic N-step approach
        result_phase, average_intensity, modulated_intensity = calculate_n_step_phase(images, phase_shifts)
    elif phase_shifting_type == PhaseShiftingAlgorithm.double_three_step:
        # Double three-step approach - average of two 3-step phases (second shifted by PI/3) 
        # Calculate formula (26-31) from section 3.2 in https://doi.org/10.1016/j.optlaseng.2018.04.019
        result_phase1, average_intensity1, modulated_intensity1 = calculate_n_step_phase(imgs[:3,:,:], phase_shifts[:3])
        result_phase2, average_intensity2, modulated_intensity2 = calculate_n_step_phase(imgs[3:,:,:], phase_shifts[3:])
        
        result_phase = (result_phase1 + result_phase2) / 2
        average_intensity = (average_intensity1 + average_intensity2) / 2
        modulated_intensity = (modulated_intensity1 + modulated_intensity2) / 2
    
    return result_phase, average_intensity, modulated_intensity


def calculate_unwraped_phase(phase_l: np.ndarray, phase_h: np.ndarray, lamb_l:float , lamb_h: float) -> np.ndarray:
    '''
    Calculate unwrapped phase from two sets of Phase Shifting Profilometry images by 
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


def calculate_phase_for_fppmeasurement(measurement: FPPMeasurement):
    '''
    Calculate unwrapped phase for FPP measurement instance with the help
    of calculate_phase_generic and calculate_unwraped_phase functions.    
    Calculated phase fields will be stored in input measurement argument.

    Args:
        measurement (FPPMeasurement): FPP measurement instance
    '''
    # Load measurement data
    shifts_count = len(measurement.shifts)
    frequencies = measurement.frequencies
    frequency_counts = len(measurement.frequencies)

    for res in measurement.camera_results:
        phases = []
        unwrapped_phases = []
        avg_ints = []
        mod_ints = []
        images = res.imgs_list

        for i in range(frequency_counts):
            images_for_one_frequency = []

            # if images is None:
            #     for j in range(shifts_count):
            #         im = load_image(measurement.imgs_file_names[i][j])
            #         images_for_one_frequency.append(im)
            # else:
            images_for_one_frequency = images[i]

            phase, avg_int, mod_int = calculate_phase_generic(images_for_one_frequency, measurement.shifts, measurement.frequencies[i], phase_shifting_type=measurement.phase_shifting_type)

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

        res.phases = phases
        res.unwrapped_phases = unwrapped_phases
        res.average_intensities = avg_ints
        res.modulated_intensities = mod_ints
    

def point_inside_polygon(x: int, y: int, poly: list[tuple(int, int)] , include_edges: bool = True) -> bool:
    '''
    Test if point (x,y) is inside polygon poly

    Point is inside polygon if horisontal beam to the right
    from point crosses polygon even number of times. Works fine for non-convex polygons.
    From: https://stackoverflow.com/questions/39660851/deciding-if-a-point-is-inside-a-polygon

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


def triangulate_points(calibration_data: dict, image1_points: np.ndarray, image2_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    '''
    Triangulate two set of 2D point in one set of 3D points

    Args:
        calibration_data (dictionary): calibration data used for triangulating
        image1_points (numpy arrray [N, 2]): first set of 2D points
        image2_points (numpy arrray [N, 2]): second set of 2D points
    Returns:
        points_3d (numpy arrray [N, 3]): triangulated 3D points
        rms1 (float): overall reprojection error for first camera
        rms2 (float): overall reprojection error for second camera
        reproj_err1, reproj_err2 (numpy arrray [N]): reprojected error for each triangulated point for first and second camera
    '''
    # Calculate the projective matrices according to the stereo calibration data
    cam1_mtx = np.array(calibration_data['camera_0']['mtx'])
    cam2_mtx = np.array(calibration_data['camera_1']['mtx'])
    dist1_mtx = np.array(calibration_data['camera_0']['dist'])
    dist2_mtx = np.array(calibration_data['camera_1']['dist'])

    # Calculate projective matrices for cameras
    proj_mtx_1 = np.dot(cam1_mtx, np.hstack((np.identity(3), np.zeros((3,1)))))
    proj_mtx_2 = np.dot(cam2_mtx, np.hstack((calibration_data['R'], calibration_data['T'])))

    # Undistort 2d points
    points_2d_1 = np.array(image1_points, dtype=np.float32)
    points_2d_2 = np.array(image2_points, dtype=np.float32)
    undist_points_2d_1 = cv2.undistortPoints(points_2d_1, cam1_mtx, dist1_mtx, P=cam1_mtx)
    undist_points_2d_2 = cv2.undistortPoints(points_2d_2, cam2_mtx, dist2_mtx, P=cam2_mtx)

    # Calculate the triangulation of 3D points
    points_hom = cv2.triangulatePoints(proj_mtx_1, proj_mtx_2, undist_points_2d_1, undist_points_2d_2)
    points_3d = cv2.convertPointsFromHomogeneous(points_hom.T)

    points_3d = np.reshape(points_3d, (points_3d.shape[0], points_3d.shape[2]))

    # Reproject triangulated points
    reproj_points, _ = cv2.projectPoints(points_3d, np.identity(3), np.zeros((3,1)), cam1_mtx, dist1_mtx)
    reproj_points2, _ = cv2.projectPoints(points_3d, np.array(calibration_data['R']), np.array(calibration_data['T']), cam2_mtx, dist2_mtx)

    # Calculate reprojection error
    reproj_err1 = np.sum(np.square(points_2d_1[:,np.newaxis,:] - reproj_points), axis=2)
    rms1 = np.sqrt(np.sum(reproj_err1)/reproj_points.shape[0])

    reproj_err2 = np.sum(np.square(points_2d_2[:,np.newaxis,:] - reproj_points2), axis=2)
    rms2 = np.sqrt(np.sum(reproj_err2/reproj_points.shape[0]))

    reproj_err1 = np.reshape(reproj_err1, (reproj_err1.shape[0]))
    reproj_err2 = np.reshape(reproj_err2, (reproj_err2.shape[0]))
    
    return points_3d, rms1, rms2, reproj_err1, reproj_err2


def calculate_bilinear_interpolation_coeficients(points: tuple[tuple]) -> np.ndarray:
    '''
    Calculate coeficients for bilinear interploation of 2d data. Bilinear interpolation is defined as
    polinomal fit f(x0, y0) = a0 + a1 * x0 + a2 * y0 + a3 * x0 * y0. Equations is used from wiki:
    https://en.wikipedia.org/wiki/Bilinear_interpolation

    Args:
        points (tuple[tuple]): four elements in format (x, y, f(x, y))
        
    Returns:
        bilinear_coeficients (numpy array): four coeficients for bilinear interploation for input points
    '''
    # Sort points
    points = sorted(points)
    
    # Get x, y coordinates and values for this points
    (x1, y1, q11), (_, y2, q12), (x2, _, q21), (_, _, q22) = points
    
    # Get matrix A
    A = np.array([[x2*y2, -x2*y1, -x1*y2, x1*y1],
                  [-y2, y1, y2, -y1],
                  [-x2, x2, x1, -x1],
                  [1, -1, -1, 1]
    ])

    # Get vector B
    B = np.array([q11, q12, q21, q22])

    # Calculate coeficients for bilinear interploation
    bilinear_coeficients = (1 / ((x2 - x1) * (y2 - y1))) * A.dot(B)
    return bilinear_coeficients


def bilinear_phase_fields_approximation(p: tuple[float, float], *data: tuple) -> tuple[float, float]:
    '''
    Calculate residiuals for bilinear interploation of horizontal and vertical phase fields.
    Function is used in find_phasogrammetry_corresponding_point fsolve function.

    Args:
        p (tuple[float, float]): x and y coordinates of point in which residiual is calculated
        data (tuple): data to calculate residiuals
            - a (numpy array): four coeficients which defines linear interploation for horizontal phase field
            - b (numpy array): four coeficients which defines linear interploation for vertical phase field
            - p_h (float): horizontal phase to match in interplotated field
            - p_v (float): vertical phase to match in interplotated field

    Returns:
        res_h, res_v (tuple[float, float]): residiuals for horizontal and vertical field in point (x, y)
    '''
    x, y = p

    a, b, p_h, p_v = data 

    return (a[0] + a[1]*x + a[2]*y + a[3]*x*y - p_h,
            b[0] + b[1]*x + b[2]*y + b[3]*x*y - p_v)    


def find_phasogrammetry_corresponding_point(p1_h: np.ndarray, p1_v: np.ndarray, p2_h: np.ndarray, p2_v: np.ndarray, x: int, y: int, LUT:list[list[list[int]]]=None) -> tuple[float, float]:
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
    # Get the phase values on vertical and horizontal phase fields 
    phase_h = p1_h[y, x]
    phase_v = p1_v[y, x]

    # If LUT available calculate corresponding points with it
    if LUT is not None:
        # Get value for x, y coordinate from LUT as first approximation 
        phase_h_index = LUT[-2].index(int(np.round(phase_h)))
        phase_v_index = LUT[-1].index(int(np.round(phase_v)))
        
        cor_points = LUT[phase_v_index][phase_h_index]

        if len(cor_points) > 0 and len(cor_points) < 20:
            # Get mean value for x, y coordinate for points from LUT as second approximation
            x, y = np.mean(cor_points, axis=0)

            iter_num = 0

            # Iterate thru variants of x and y where fields are near to phase_v and phase_h
            while iter_num < 5: 
                # Get neareast coords to current values of x and y
                if int(np.round(x)) - x == 0:
                    x1 = int(x - 1)
                    x2 = int(x + 1)
                else:
                    x1 = int(np.floor(x))
                    x2 = int(np.ceil(x))

                if int(np.round(y)) - y == 0:
                    y1 = int(y - 1)
                    y2 = int(y + 1)
                else:
                    y1 = int(np.floor(y))
                    y2 = int(np.ceil(y))
  
                # Check if coords are on field (are positive and less than field shape)
                if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0 and x1 < p1_h.shape[1] and x2 < p1_h.shape[1] and y1 < p1_h.shape[0] and y2 < p1_h.shape[0]:

                    # Get coeficients for bilinear interploation for horizontal phase
                    aa = calculate_bilinear_interpolation_coeficients(((x1, y1, p2_h[y1, x1]), (x1, y2, p2_h[y2, x1]),
                                                                    (x2, y2, p2_h[y2, x2]), (x2, y1, p2_h[y2, x1])))
                    # Get coeficients for bilinear interploation for vertical phase
                    bb = calculate_bilinear_interpolation_coeficients(((x1, y1, p2_v[y1, x1]), (x1, y2, p2_v[y2, x1]),
                                                                    (x2, y2, p2_v[y2, x2]), (x2, y1, p2_v[y2, x1])))

                    # Find there bilinear interploation is equal to phase_h and phase_v
                    x, y =  fsolve(bilinear_phase_fields_approximation, (x1, y1), args=(aa, bb, phase_h, phase_v))

                    # TODO: Return residiuals from function
                    # Calculate residiuals
                    h_res, v_res = bilinear_phase_fields_approximation((x, y), aa, bb, phase_h, phase_v) 

                    # Check if x and y are between x1, x2, y1 and y2
                    if x2 > x > x1 and y2 > y > y1:
                        return x, y
                    else:
                        iter_num = iter_num + 1
                else:
                    return -1, -1

            return -1, -1
        else:
            return -1, -1

    # Find coords of isophase curves
    y_h, x_h = np.where(np.isclose(p2_h, phase_h, atol=10**-1))
    y_v, x_v = np.where(np.isclose(p2_v, phase_v, atol=10**-1))

    # Break if isoline not found
    if y_h.size == 0 or y_v.size == 0:
        return -1, -1

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


def get_phasogrammetry_correlation(p1_h: np.ndarray, p1_v: np.ndarray, p2_h: np.ndarray, p2_v: np.ndarray, x: int, y: int, window_size: int) -> np.ndarray:
    '''
    Calculate correlation function for horizontal and vertical phase fields

    Args:
        p1_h (numpy array): phase field for horizontal fringes for first camera
        p1_v (numpy array): phase field for vertical fringes for first camera
        p2_h (numpy array): phase field for horizontal fringes for second camera
        p2_v (numpy array): phase field for vertical fringes for second camera
        x (int): horizontal coordinate of point for first camera
        y (int): vertical coordinate of point for first camera
        window_size (int): size of window to calculate correlation function

    Returns:
        corelation_field (numpy array): calculated correlation field
    '''
    p1_h_ij = p1_h[int(y - window_size//2):int(y + window_size//2), int(x - window_size//2):int(x + window_size//2)]
    p1_v_ij = p1_v[int(y - window_size//2):int(y + window_size//2), int(x - window_size//2):int(x + window_size//2)]
    p1_h_m = np.mean(p1_h_ij)
    p1_v_m = np.mean(p1_v_ij)

    corelation_field = np.zeros((window_size, window_size))

    xx = np.linspace(x - window_size // 2, x + window_size // 2, window_size)
    yy = np.linspace(y - window_size // 2, y + window_size // 2, window_size)

    for j in range(yy.shape[0]):
        for i in range(xx.shape[0]):
            x0 = xx[i]
            y0 = yy[j]
            p2_h_ij = p2_h[int(y0 - window_size //2):int(y0 + window_size //2), int(x0 - window_size//2):int(x0 + window_size//2)]
            p2_v_ij = p2_v[int(y0 - window_size //2):int(y0 + window_size //2), int(x0 - window_size//2):int(x0 + window_size//2)]
            p2_h_m = np.mean(p2_h_ij)
            p2_v_m = np.mean(p2_v_ij)
            t1_h = (p1_h_ij - p1_h_m) ** 2
            t1_v = (p1_v_ij - p1_v_m) ** 2
            t2_h = (p2_h_ij - p2_h_m) ** 2
            t2_v = (p2_v_ij - p2_v_m) ** 2

            if p2_h_ij.size == p1_h_ij.size and p2_v_ij.size == p1_v_ij.size:
                t = np.sum(t1_h * t1_v * t2_h * t2_v) / np.sqrt(np.sum(t1_h * t1_v) * np.sum(t2_h * t2_v))
                # if t < 1:
                corelation_field[j, i] = t

    return corelation_field


def get_phase_field_ROI(fpp_measurement: FPPMeasurement, signal_to_nose_threshold: float = 0.25):
    '''
    Get ROI for FPP measurement with the help of signal to noise thresholding.
    ROI is stored as a mask (fpp_measurement.signal_to_noise_mask) with values 0 for points
    with signal-to-noise ratio below threshold and 1 for points with ratio above threshold.
    Additionally ROI is stored as a quadrangle defined by four points consisting of minimum
    and maximum x and y coordinates for points with signal-to-noise ratio above the threshold.
    Calculated ROI will be stored in input fpp_measurement argument.

    Args:
        fpp_measurement (FPPMeasurement): FPP measurment for calcaulating ROI
        signal_to_nose_threshold (float) = 0.25: threshold for signal to noise ratio to calcaulate ROI
    '''
    for res in fpp_measurement.camera_results:
        # Calculate signal to noise ratio
        signal_to_nose = res.modulated_intensities[-1] / res.average_intensities[-1]
        # Threshold signal to noise with defined threshold level
        thresholded_coords = np.argwhere(signal_to_nose > signal_to_nose_threshold)

        # Store ROI mask
        res.signal_to_noise_mask = np.zeros(signal_to_nose.shape, dtype=int)
        res.signal_to_noise_mask[signal_to_nose > signal_to_nose_threshold] = 1

        # Determine four points around thresholded area
        x_min = np.min(thresholded_coords[:,1])
        x_max = np.max(thresholded_coords[:,1])
        y_min = np.min(thresholded_coords[:,0])
        y_max = np.max(thresholded_coords[:,0])

        # Store determined ROI
        res.ROI = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])


def get_phase_field_LUT(fpp_measurement_h: FPPMeasurement, fpp_measurement_v: FPPMeasurement) -> list[list[list]]:
    '''
    Get LUT for horizontal and vertical phase field to increase phasogrammetry calculation speed.
    LUT is a two-dimensional array of coordinates whose indices correspond to the values of horizontal
    and vertical phase in these coordinates. Knowing the values of the horizontal and vertical phase,
    you can quickly find the coordinates of points with these values.
    The LUT is a list of lists of lists of two-dimensional coordinates.

    Args:
        fpp_measurement_h (FPPMeasurement): FPP measurment for horizontal fringes
        fpp_measurement_v (FPPMeasurement): FPP measurment for vertical fringes
    Returns:
        LUT (list[list[list]]): LUT structure containing the coordinates of points for the horizontal and vertical phase values
    '''
    p_h = fpp_measurement_h.unwrapped_phases[-1]
    p_v = fpp_measurement_v.unwrapped_phases[-1]

    # Find range for horizontal and vertical phase
    ph_max = np.max(p_h)
    ph_min = np.min(p_h)
    pv_max = np.max(p_v)
    pv_min = np.min(p_v)
    h_range = np.arange(ph_min, ph_max)
    v_range = np.arange(pv_min, pv_max)

    # Determine size of LUT structure
    w, h = h_range.shape[0] + 1, v_range.shape[0] + 1

    # Create LUT structure
    LUT = [[[] for x in range(w)] for y in range(h)]

    w = p_h.shape[1]
    h = p_h.shape[0]

    # Phase rounding with an offset so that they start from zero  
    p_h_r = np.round(p_h - ph_min).astype(int).tolist()
    p_v_r = np.round(p_v - pv_min).astype(int).tolist()

    # Fill LUT with coordinates of points with horizontal and vertical values as indicies
    for y in range(h):
        for x in range(w):
            if fpp_measurement_h.signal_to_noise_mask[y, x] == 1:
                LUT[p_v_r[y][x]][p_h_r[y][x]].append([x, y])
    
    # Add range of horizontal and vertical phases at the end of LUT
    LUT.append(np.round(h_range).astype(int).tolist())
    LUT.append(np.round(v_range).astype(int).tolist())

    return LUT


def process_fppmeasurement_with_phasogrammetry(measurement: FPPMeasurement, step_x: float, step_y: float, LUT:list[list[list[int]]]=None) -> tuple[np.ndarray, np.ndarray]:
    '''
    Find 2D corresponding points for two phase fields sets with phasogrammetry approach 

    Args:
        fppmeasurements_h (list of FPPMeasurement): list of FPPMeasurements instances with horizontal fringes
        fppmeasurements_v (list of FPPMeasurement): list of FPPMeasurements instances with vertical fringes
        step_x, step_y (float): horizontal and vertical steps to calculate corresponding points
        LUT (list[list[list]]): LUT structure containing the coordinates of points for the horizontal and vertical phase values
    Returns:
        points_1 (numpy array [N, 2]): corresponding 2D points from first camera
        points_2 (numpy array [N, 2]): corresponding 2D points from second camera
    '''
    # Take phases with highest frequencies 
    p1_h = measurement.camera_results[2].unwrapped_phases[-1]
    p2_h = measurement.camera_results[3].unwrapped_phases[-1]

    p1_v = measurement.camera_results[0].unwrapped_phases[-1]
    p2_v = measurement.camera_results[1].unwrapped_phases[-1]

    # Get ROI from measurement object
    ROI1 = measurement.camera_results[0].ROI

    # Cut ROI from phase fields for second camera
    ROIx = slice(0, measurement.camera_results[1].unwrapped_phases[-1].shape[1])
    ROIy = slice(0, measurement.camera_results[1].unwrapped_phases[-1].shape[0])

    p2_h = p2_h[ROIy, ROIx]
    p2_v = p2_v[ROIy, ROIx]

    # Calculation of the coordinate grid on first image
    xx = np.arange(0, p1_h.shape[1], step_x, dtype=np.int32)
    yy = np.arange(0, p1_h.shape[0], step_y, dtype=np.int32)

    coords1 = []

    for y in yy:
        for x in xx:
            # Check if coordinate in ROI rectangle
            # if measurements_h[0].signal_to_noise_mask[y, x] == 1:
            #   coords1.append((x, y))
            if point_inside_polygon(x, y, ROI1):
                coords1.append((x, y))

    coords2 = []

    coords_to_delete = []

    if config.USE_MULTIPROCESSING:
        # Use parallel calaculation to increase processing speed 
        with multiprocessing.Pool(config.POOLS_NUMBER) as p:
            coords2 = p.starmap(find_phasogrammetry_corresponding_point, [(p1_h, p1_v, p2_h, p2_v, coords1[i][0], coords1[i][1], LUT) for i in range(len(coords1))])

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
            x, y = find_phasogrammetry_corresponding_point(p1_h, p1_v, p2_h, p2_v, coords1[i][0], coords1[i][1], LUT)
            # If no point found, delete coordinate from grid
            if x == -1 and y == -1:
                coords_to_delete.append(i)
            else:
                coords2.append((x + ROIx.start, y + ROIy.start))

        # Delete point in grid with no coresponding point on second image
        for index in reversed(coords_to_delete):
            coords1.pop(index)

    # Form a set of coordinates of corresponding points on the first and second images
    image1_points = []
    image2_points = []
    distance = []

    for point1, point2 in zip(coords1, coords2):
        image1_points.append([point1[0], point1[1]]) 
        image2_points.append([point2[0], point2[1]])
        distance.append(((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5)

    # Remove outliers
    std_d = np.std(distance)
    indicies_to_delete = [i for i in range(len(distance)) if distance[i] > std_d*10]
    for index in reversed(indicies_to_delete):
        image1_points.pop(index)
        image2_points.pop(index)

    # Convert list to array before returning result from function
    image1_points = np.array(image1_points, dtype=np.float32)
    image2_points = np.array(image2_points, dtype=np.float32)

    return image1_points, image2_points


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