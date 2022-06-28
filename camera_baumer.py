from camera import Camera

class CameraBaumer(Camera):
    def __init__(self, baumer_cam_object) -> None:
        baumer_cam_object.Connect()
        self.__cam_baumer_ref = baumer_cam_object
        self.type = 'baumer'
        # self.exposure = baumer_cam_object.f.ExposureTime.value
        # self.gain = baumer_cam_object.f.Gain.value
        # self.gamma = baumer_cam_object.f.Gamma.value
    
    @property
    def exposure(self):
        return self.__cam_baumer_ref.f.ExposureTime.value
    
    @exposure.setter
    def exposure(self, x):
        self.__cam_baumer_ref.f.ExposureTime.value = x
    
    @property
    def gain(self):
        return self.__cam_baumer_ref.f.Gain.value
    
    @gain.setter
    def gain(self, x):
        self.__cam_baumer_ref.f.Gain.value = x
    
    @property
    def gamma(self):
        return self.__cam_baumer_ref.f.Gamma.value
    
    @gamma.setter
    def gamma(self, x):
        self.__cam_baumer_ref.f.Gamma.value = x

    def get_image(self):
        return self.__cam_baumer_ref.GetImage().GetNPArray()