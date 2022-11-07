'''Module to store FPP data structures'''

from __future__ import annotations
from typing import Optional

import enum
from dataclasses import dataclass, field

import numpy as np


class PhaseShiftingAlgorithm(enum.IntEnum):
    '''Enum for phase shift algorithm type'''
    n_step = 1
    double_three_step = 2


@dataclass
class CameraMeasurement:
    '''
    Class to store result of measurement for one camera
    '''
    fringe_orientation: Optional[str] = 'vertical'    
    imgs_list: Optional[list[np.ndarray]] = field(default_factory=lambda:list())
    imgs_file_names: Optional[list[str]] = field(default_factory=lambda:list())

    # Calculated attributes
    phases: Optional[list[np.ndarray]] = field(init=False)
    unwrapped_phases: Optional[list[np.ndarray]] = field(init=False)
    average_intensities: Optional[list[np.ndarray]] = field(init=False)
    modulated_intensities: Optional[list[np.ndarray]] = field(init=False)
    signal_to_noise_mask: Optional[np.ndarray] = field(init=False)
    ROI: Optional[np.array] = field(init=False)


@dataclass
class FPPMeasurement:
    '''
    Class to store FPP measurement data    
    '''
    phase_shifting_type: PhaseShiftingAlgorithm
    frequencies: list[float]
    shifts: list[float]

    camera_results: list[CameraMeasurement] = field(default_factory=lambda:list())

    @property
    def frequency_counts(self) -> int:
        return len(self.frequencies)

    @property
    def shifts_count(self) -> int:
        return len(self.shifts)