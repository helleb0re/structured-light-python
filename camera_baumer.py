from __future__ import annotations

import numpy as np

try:    
    import neoapi
except ImportError:
    neoapi_found = False
else:
    neoapi_found = True

from camera import Camera


class CameraBaumer(Camera):
    def __init__(self, camera : neoapi.Cam):
        self.camera = camera
        self.camera.Connect()
        self.type = 'baumer'
        # print(f'Camera {camera.f.DeviceModelName.value} {camera.f.DeviceSerialNumber.value}')

    @staticmethod
    def get_available_cameras(cameras_num_to_find:int=2) -> list[Camera]:
        cameras = []
        
        if neoapi_found:
            for i in range(cameras_num_to_find):
                camera = CameraBaumer(neoapi.Cam())

                # Set default cameras parameters
                camera.exposure = 80000
                camera.gain = 4
                camera.frame_rate_enable = True
                camera.frame_rate = 25.0

                # Set first camera as master for triggering
                if i == 0:
                    camera.trigger_mode = neoapi.TriggerMode_Off
                    camera.line_selector = neoapi.LineSelector_Line1
                    camera.line_mode = neoapi.LineMode_Output
                    camera.line_source = neoapi.LineSource_ExposureActive
                
                # Set next camera as slave for trigger wait
                if i != 0:
                    camera.trigger_mode = neoapi.TriggerMode_On
                
                cameras.append(camera)

        return cameras

    def get_image(self) -> np.array:
        img = self.camera.GetImage().GetNPArray()
        return img.reshape(img.shape[0], img.shape[1])

    @property
    def exposure(self):
        return self.camera.f.ExposureTime.value

    @exposure.setter
    def exposure(self, x):
        self.camera.f.ExposureTime.value = x

    @property
    def gain(self):
        return self.camera.f.Gain.value

    @gain.setter
    def gain(self, x):
        self.camera.f.Gain.value = x

    @property
    def gamma(self):
        return self.camera.f.Gamma.value

    @gamma.setter
    def gamma(self, x):
        self.camera.f.Gamma.value = x

    @property
    def exposure_auto(self):
        return self.camera.f.ExposureAuto.value

    @exposure_auto.setter
    def exposure_auto(self, x):
        self.camera.f.ExposureAuto.value = x

    @property
    def trigger_mode(self):
        return self.camera.f.TriggerMode.value
    
    @trigger_mode.setter
    def trigger_mode(self, x):
        self.camera.f.TriggerMode.value = x
    
    @property
    def line_selector(self):
        return self.camera.f.LineSelector.value

    @line_selector.setter
    def line_selector(self, x):
        self.camera.f.LineSelector.value = x

    @property
    def line_mode(self):
        return self.camera.f.LineMode.value

    @line_mode.setter
    def line_mode(self, x):
        self.camera.f.LineMode.value = x

    @property
    def line_source(self):
        return self.camera.f.LineSource.value

    @line_source.setter
    def line_source(self, x):
        self.camera.f.LineSource.value = x

    @property
    def frame_rate_enable(self):
        return self.camera.f.AcquisitionFrameRateEnable.value

    @frame_rate_enable.setter
    def frame_rate_enable(self, x):
        self.camera.f.AcquisitionFrameRateEnable.value = x

    @property
    def frame_rate(self):
        return self.camera.f.AcquisitionFrameRate.value
    
    @frame_rate.setter
    def frame_rate(self, x):
        self.camera.f.AcquisitionFrameRate.value = x
