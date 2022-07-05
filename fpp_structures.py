'''Module to store FPP data structures'''

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class FPPMeasurement:
    '''Class to store FPP measurement data'''
    frequencies : list[float]
    shifts : list[float]
    imgs_file_names : list[list[str]]
    imgs_list: Optional[list[list[np.array]]] = None

    @property
    def frequency_counts(self) -> int :
        return len(self.frequencies)

    @property
    def shifts_count(self) -> int :
        return len(self.shifts)