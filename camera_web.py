import cv2
from camera import Camera


class CameraWeb(Camera):

    def __init__(self, width=1920, height=1080, number = 0):
        self.camera = cv2.VideoCapture(number, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.type = 'web'

    def get_image(self):
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

    
    #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    #cap.set(cv2.CAP_PROP_AUTO_WB, 1)