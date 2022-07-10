'''Module to process FPP images'''

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt

from create_patterns import *
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

        result_phase = np.arctan2(np.sum(temp1, 0), np.sum(temp2, 0))

        # Calculate formula (9-10) in https://doi.org/10.1016/j.optlaseng.2018.04.019
        average_intensity = np.mean(imgs, 0) / len(images)
        modulated_intensity = 2 * np.sqrt(np.power(np.sum(temp1, 0), 2) + np.power(np.sum(temp2, 0), 2)) / len(images)

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

    for i in range(frequency_counts):

        images_for_one_frequency = []

        if images is None:
            for j in range(shifts_count):
                im = load_image(measurement.imgs_file_names[i][j])
                images_for_one_frequency.append(im)
        else:
            images_for_one_frequency = images[i]

        phase, avg_int, mod_int = calculate_phase_generic(images_for_one_frequency, measurement.shifts, measurement.frequencies[i])
        phases.append(phase)

        if i == 0:
            unwrapped_phases.append(phase)
        else:
            unwrapped_phase = calculate_unwraped_phase(unwrapped_phases[i-1], phases[i], 1 / frequencies[i-1], 1 / frequencies[i])
            unwrapped_phases.append(unwrapped_phase)
    
    return phases, unwrapped_phases