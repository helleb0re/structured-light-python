'''Module to store FPP data structures'''

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
import enum
import numpy as np

class PhaseShiftingAlgorithm(enum.IntEnum):
    n_step = 1
    double_three_step = 2


@dataclass
class FPPMeasurement:
    '''Class to store FPP measurement data'''
    frequencies: list[float]
    shifts: list[float]
    imgs_file_names: list[list[str]]
    phases: Optional[list[np.ndarray]] = field(init=False)
    unwrapped_phases: Optional[list[np.ndarray]] = field(init=False)
    average_intensities: Optional[list[np.ndarray]] = field(init=False)
    modulated_intensities: Optional[list[np.ndarray]] = field(init=False)
    signal_to_noise_mask: Optional[np.ndarray] = field(init=False)
    fringe_orientation: Optional[str] = 'vertical'
    imgs_list: Optional[list[list[np.ndarray]]] = None
    phase_shifting_type: PhaseShiftingAlgorithm = field(init=False)

    @property
    def frequency_counts(self) -> int:
        return len(self.frequencies)

    @property
    def shifts_count(self) -> int:
        return len(self.shifts)