from __future__ import annotations

import cv2
import numpy as np

from camera import Camera


class CameraWeb(Camera):

    def __init__(self, width=1920, height=1080, number = 0):
        self.camera = cv2.VideoCapture(number, cv2.CAP_DSHOW)
        if self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.type = 'web'
        else:
            # If camera is not opened, it is used in another place
            raise ValueError()
    
    @staticmethod
    def get_available_cameras(cameras_num_to_find:int=2) -> list[Camera]:
        cameras = []
        # Try to find defined number of cameras
        for i in range(cameras_num_to_find):
            try:
                cameras.append(CameraWeb(number=i))
            except ValueError:
                # Skip camera with i id
                pass
        return cameras

    def get_image(self) -> np.array:
        if self.camera.isOpened:
            return self.camera.read()[1]

    @property 
    def focus(self):  
        return self.camera.get(cv2.CAP_PROP_FOCUS)
        
    @focus.setter  
    def focus(self, x):  
        self.camera.set(cv2.CAP_PROP_FOCUS, x)

    @property
    def exposure(self):  
        return self.camera.get(cv2.CAP_PROP_EXPOSURE)
        
    @exposure.setter
    def exposure(self, x):  
        self.camera.set(cv2.CAP_PROP_EXPOSURE, x)

    @property
    def brightness(self):
        return self.camera.get(cv2.CAP_PROP_BRIGHTNESS)

    @brightness.setter
    def brightness(self, x):
        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, x)

    @property
    def gamma(self):
        return self.camera.get(cv2.CAP_PROP_GAMMA)

    @gamma.setter
    def gamma(self, x):
        self.camera.set(cv2.CAP_PROP_GAMMA, x)

    @property
    def gain(self):
        raise NotImplemented()

    @gain.setter
    def gain(self):
        raise NotImplemented()
    
    #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    #cap.set(cv2.CAP_PROP_AUTO_WB, 1)