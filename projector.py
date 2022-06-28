import cv2
import numpy as np

class Projector():

    def __init__(self, width, height, min_brightness = 0, max_brightness = 255):
        self.width = width
        self.height = height
        self.__min_image_brightness = min_brightness
        self.__max_image_brightness = max_brightness
        self.window_exist = False

    def set_up_window(self):
        cv2.namedWindow('projection', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('projection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('projection', 1920, 0)
        self.window_exist = True

    def project_pattern(self, pattern):
        if not self.window_exist:
            self.set_up_window()
        pat = self.image_brightness_rescale_factor * (pattern + 1) + self.min_image_brightness
        cv2.imshow('projection', pat.astype(np.uint8))

    def close_window(self):
        cv2.destroyWindow('projection')
        self.window_exist = False

    @property
    def resolution(self):
        return self.width, self.height

    @property
    def min_image_brightness(self):
        return self.__min_image_brightness

    @min_image_brightness.setter
    def min_image_brightness(self, value):
        self.__min_image_brightness = value

    @property
    def max_image_brightness(self):
        return self.__max_image_brightness

    @max_image_brightness.setter
    def max_image_brightness(self, value):
        self.__max_image_brightness = value

    @property
    def image_brightness_rescale_factor(self):
        return (self.max_image_brightness - self.min_image_brightness) / 2

