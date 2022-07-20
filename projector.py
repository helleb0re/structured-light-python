'''Module for Projector class'''
from __future__ import annotations

import cv2
import numpy as np

import config


class Projector():
    '''
    Class to control projector during experiment
    '''
    def __init__(self, width: int, height: int, min_brightness: float = 0, max_brightness: float = 255):
        self.width = width
        self.height = height
        self.__min_image_brightness = min_brightness
        self.__max_image_brightness = max_brightness
        self.window_exist: bool = False

    def set_up_window(self) -> None:
        '''
        Open new window thru OpenCV GUI and show it on second extended screen
        '''
        cv2.namedWindow('Projector window', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Projector window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # TODO: remove magic number 1920 with screen resolution for multimonitor case
        # https://stackoverflow.com/questions/3129322/how-do-i-get-monitor-resolution-in-python
        cv2.moveWindow('Projector window', 1920, 0)
        self.window_exist = True

    def project_pattern(self, pattern: np.ndarray, correction: bool = True) -> None:
        '''
        Project pattern thru OpenCV GUI window, before projection pattern is intensity corrected

        Args:
            pattern (numpy array): image to project with OpenCV imshow method
            correction (bool): do intensity correction before project pattern
        '''
        # Open OpenCV GUI window, if it is has not been already opened
        if not self.window_exist:
            self.set_up_window()
        
        # Correct image with calibration coefficients
        if correction:
            # self._corrected_pattern = self.image_brightness_rescale_factor * (pattern + 1) + self.min_image_brightness
            self._corrected_pattern = ((pattern - config.PROJECTOR_GAMMA_C) / config.PROJECTOR_GAMMA_A) ** (1 / config.PROJECTOR_GAMMA_B)
        else:
            self._corrected_pattern = pattern
        # Show image at OpenCV GUI window
        cv2.imshow('Projector window', self._corrected_pattern)

    def close_window(self) -> None:
        '''
        Close opened OpenCV GUI window on second extended screen
        '''
        cv2.destroyWindow('Projector window')
        self.window_exist = False

    @property
    def corrected_pattern(self) -> np.ndarray:
        '''
        Return last projected corrected pattern as numpy array
        '''
        return self._corrected_pattern

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def min_image_brightness(self) -> float:
        return self.__min_image_brightness

    @min_image_brightness.setter
    def min_image_brightness(self, value: float):
        self.__min_image_brightness = value

    @property
    def max_image_brightness(self) -> float:
        return self.__max_image_brightness

    @max_image_brightness.setter
    def max_image_brightness(self, value: float):
        self.__max_image_brightness = value

    @property
    def image_brightness_rescale_factor(self) -> float:
        return (self.max_image_brightness - self.min_image_brightness) / 2
