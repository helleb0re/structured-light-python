from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty

import numpy as np


class Camera(ABC):

    @staticmethod
    @abstractmethod
    def get_available_cameras(cameras_num_to_find:int=2) -> list[Camera]:
        '''
        Get list of available cameras
        
        Args:
            cameras_num_to_find (int) = 2 : The number of cameras to try to find 

        Returns:
            cameras (list of Camera): list of founded cameras
        '''

    @abstractmethod
    def get_image(self) -> np.array:
        '''
        Get image from camera
        
        Returns:
            image (numpy array): image as numpy array (2D or 3D depending on color mode)
        '''
    
    @abstractproperty
    def exposure(self):
        '''Exposure'''
    
    @exposure.setter
    @abstractmethod
    def exposure(self):
        '''Set exposure'''

    @abstractproperty
    def gain(self):
        '''Gain'''

    @gain.setter
    @abstractmethod
    def gain(self):
        '''Set gain'''
    
    @abstractproperty
    def gamma(self):
        '''Gamma'''
    
    @gamma.setter
    @abstractmethod
    def gamma(self):
        '''Set gamma'''
