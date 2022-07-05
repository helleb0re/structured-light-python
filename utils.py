'''Module to store utils for working with data'''

from __future__ import annotations

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


files_path = './frames_21_06/cam_1/'
img_mask = 'frame_{0}_{1}.png'
shifts_count = 4
frequencies = [1 + i* 3 for i in range(7)] 

measurement = create_fpp_measurement_from_files(files_path, img_mask, shifts_count, frequencies)

phases, unwrapped_phases = calculate_phase_for_fppmeasurement(measurement)

for phase in unwrapped_phases:
    plt.imshow(phase, cmap='gray')
    plt.colorbar()
    plt.show()
