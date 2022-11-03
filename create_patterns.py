'''Module for creating FPP patterns'''

from __future__ import annotations
from typing import Union

import numpy as np

from fpp_structures import PhaseShiftingAlgorithm


def create_psp_template(width: int, height: int, frequency: float, shifts_number: int = 4, vertical: bool = True, delta_fi: float = 0) -> tuple[list[np.ndarray], list[float]]:
    '''
    Create set of patterns for phase shift profilometry for one frequencies.
    Patterns returned as list of numpy arrays.

    Args:
        width (int): width of patterns to generate
        height (int): height of patterns to generate
        frequency (float): frequency of patterns to generate
        shifts_number (int): number of phase shifts for generated patterns
        vertical (bool): create vertical fringes, if False create horizontal

    Returns:
        patterns (list[np.ndarray]): list of generated patterns
        phase_shifts (list[float]): list of phase shifts for generated patterns
    '''
    patterns = []

    # Determine length of cos sequence
    if vertical:
        length = width
    else:
        length = height
    
    # Create x sequence
    x = np.linspace(0, length, length)

    # Calculate phase shifts
    phase_shifts = [2 * np.pi / shifts_number * i + delta_fi for i in range(shifts_number)]
    
    for phase_shift in phase_shifts:
        # Calculate cos sequence with defined parameters
        cos = 0.5 + 0.5 * np.cos(2 * np.pi * frequency * (x / length) - phase_shift)
        
        # Tile cos sequence for vertical or horizontal fringe orientation
        if vertical:
            pattern = np.tile(cos, (height, 1))
        else:
            pattern = np.tile(cos.reshape((-1, 1)), (1, width))
        
        patterns.append(pattern)

    return patterns, phase_shifts

def create_psp_templates(width: int, height: int, frequencies: Union[int, list[float]], phase_shift_type: PhaseShiftingAlgorithm, shifts_number: int = 4, vertical: bool = True) -> tuple[list[list[np.ndarray]], list[float], list[float]]:
    '''
    Create set of patterns for phase shift profilometry for several frequencies.
    Patterns returned as list of list of numpy arrays.
    Outer list contains list with shifts_numbers patterns for each frequency.
    Patterns for one frequency generated via create_psp_template().

    Args:
        width (int): width of patterns to generate
        height (int): height of patterns to generate
        frequencies (int or list[float]): frequencies (number or list) of patterns to generate
        shifts_number (int): number of phase shifts for one frequency
        vertical (bool): create vertical patterns, if False create horizontal

    Returns:
        patterns (list[list[np.ndarray]]): set of generated patterns
        phase_shifts (list[float]): list of phase shifts for generated patterns
        frequencies (list[float]): list of frequencies for generated pattern sets
    '''
    patterns = []

    if phase_shift_type == PhaseShiftingAlgorithm.n_step:
        for frequency in frequencies:
            template, phase_shifts = create_psp_template(width, height, frequency, shifts_number, vertical)
            patterns.append(template)
    elif phase_shift_type == PhaseShiftingAlgorithm.double_three_step:
        for frequency in frequencies:
            template, phase_shifts = create_psp_template(width, height, frequency, 3, vertical)
            patterns.append(template)
            template, phase_shifts2 = create_psp_template(width, height, frequency, 3, vertical, -np.pi / 3)
            patterns[-1].extend(template)
            phase_shifts.extend(phase_shifts2)

    return patterns, phase_shifts, frequencies 

def linear_gradient(width: int, height: int, vertical: bool = True) -> np.ndarray:
    '''
    Create linear gradient pattern. It can be used for calibration purpose.

    Args:
        width (int): width of patterns to generate
        height (int): height of patterns to generate
        vertical (bool): create vertical gradient, if False create horizontal

    Returns:
        gradient (np.ndarray): generated linear gradient pattern
    '''
    # Determine length of cos sequence
    if vertical:
        length = width
    else:
        length = height
    
    # Create x sequence
    x = np.linspace(0, length, length)

    # Calculate gradient sequence
    gradient = x / length

    # Tile gradient sequence for vertical or horizontal orientation
    if vertical:
        gradient = np.tile(gradient, (height, 1))
    else:
        gradient = np.tile(gradient.reshape((-1, 1)), (1, width))

    return gradient