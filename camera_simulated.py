'''Module for simulated camera class for test purpose'''

from __future__ import annotations

import numpy as np

from camera import Camera
from projector import Projector


class CameraSimulated(Camera):
    def __init__(self):        
        self.type = 'simulated'
        self._projector = None
        print(f'Simulated camera created')

    @staticmethod
    def get_available_cameras(cameras_num_to_find:int=2) -> list[Camera]:
        cameras = []
        
        for _ in range(cameras_num_to_find):
            cameras.append(CameraSimulated())

        return cameras

    def get_image(self) -> np.array:
        if self.projector is not None:
            img = self._projector.corrected_pattern
            return img
        else:
            raise ValueError()

    @property
    def projector(self) -> Projector:
        return self._projector

    @projector.setter
    def projector(self, projector: Projector):
        self._projector = projector

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, x):
        self._exposure = x

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, x):
        self._gain = x

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, x):
        self._gamma = x
