from __future__ import annotations

import neoapi

from camera import Camera

class CameraBaumer(Camera):
    def __init__(self, camera : neoapi.Cam) -> None:
        self.camera = camera
        self.camera.Connect()
        self.type = 'baumer'
        # print(f'Camera {camera.f.DeviceModelName.value} {camera.f.DeviceSerialNumber.value}')

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

    def get_image(self):
        img = self.camera.GetImage().GetNPArray()
        return img.reshape(img.shape[0], img.shape[1])