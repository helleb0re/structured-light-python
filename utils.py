'''Module to store utils for working with data'''

from __future__ import annotations
from typing import Optional

import json

import numpy as np
from matplotlib import pyplot as plt

from fpp_structures import FPPMeasurement
from processing import calculate_phase_for_fppmeasurement


def create_fpp_measurement_from_files(files_path : str, file_mask : str, shifts_count : int, frequencies : list[float]) -> FPPMeasurement:
    '''
    Create FPPMeasurement instance from image files
    
    Args:
        files_path (str): The path to images files 
        file_mask (str): The file mask which used for files itterating
        thru phase shifts and frequeincies, e.i. 'frame_{0}_{1}.png'. First argument
        in mask is frequeincy number, second is phase shift number
        shifts_count (int): The count of phase shifts used during images capturing
        frequencies (list): The list of frequencies used during images capturing

    Returns:
        measurement (FPPMeasurement): FPPMeasurement instance
    '''

    # Filenames for image files
    filenames = []

    for i in range(len(frequencies)):
        # List of filenames for one frequency
        one_frequency_files = []
        for j in range(shifts_count):
            one_frequency_files.append(files_path + file_mask.format(i, j))  
        filenames.append(one_frequency_files) 

    # Calculate phase shifts
    shifts = [2 * np.pi / len(shifts_count) * i for i in range(shifts_count)]

    # Create new FPPMeasurement instance
    measurement = FPPMeasurement(
        shifts = shifts,
        frequencies = frequencies,
        imgs_file_names = filenames,
    )
    
    return measurement


def load_fpp_measurements(file: str) -> list[FPPMeasurement]:
    '''
    Load FPPMeasurements from json file
    
    Args:
        file (str): the path to FPPMeasurements json file 

    Returns:
        measurements (list[FPPMeasurement]): list of FPPMeasurement instances loaded from file
    '''
    measurements = []

    with open(file, 'r') as fp:
        data = json.load(fp)

    for instance in data:
        # Create new FPPMeasurement instance
        measurement = FPPMeasurement(
            shifts = instance['shifts'],
            frequencies = instance['frequencies'],
            fringe_orientation = instance['fringe_orientation'],
            
        )

        measurement.imgs_file_names = instance['imgs_file_names']

        measurements.append(measurement)

    return measurements 


def get_quiverplot(coords: np.ndarray, maximums: np.ndarray, image: Optional[np.ndarray] = None, stretch_factor: float = 5.0):
    """Draw a vector field of displacements on input image (if defined)

        Args:
            coords (list[np.ndarray]): list of vectors start points
            maximums (list[np.ndarray]): list of vectors end points
            image=None (np.ndarray): image to draw vector field on
            stretch_factor=5.0 (float): magnitude factor for vector length
    """      
    x = [coord[0] for coord in coords]
    y = [coord[1] for coord in coords]
    u = [maximum[0] * stretch_factor for maximum in maximums]
    v = [maximum[1] * stretch_factor for maximum in maximums]
    c = [maximum[2] for maximum in maximums]

    plt.quiver(x, y, u, v, c, scale=1, units='xy', cmap='jet')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Normal correlation function maximum')

    plt.xlabel('X, pixels')
    plt.ylabel('Y, pixels')
        
    if image is not None:
        plt.imshow(image, cmap='gray')
     
    plt.show()
