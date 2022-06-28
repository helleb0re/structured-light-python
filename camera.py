from abc import ABC, abstractmethod, abstractproperty

class Camera(ABC):
    # @abstractproperty
    # def focus(self):  
    #     """Focus"""
    
    # @focus.setter
    # @abstractmethod
    # def focus(self, x):
    #     """Set focus"""
    
    @abstractproperty
    def exposure(self):
        """Exposure"""
    
    @exposure.setter
    @abstractmethod
    def exposure(self):
        """Set exposure"""

    @abstractproperty
    def gain(self):
        """Gain"""

    @gain.setter
    @abstractmethod
    def gain(self):
        """Set gain"""
    
    # @abstractproperty
    # def brightness(self):
    #     """Brightness"""
    
    # @brightness.setter
    # @abstractmethod
    # def brightness(self):
    #     """Set brightness"""
    
    @abstractproperty
    def gamma(self):
        """Gamma"""
    
    @gamma.setter
    @abstractmethod
    def gamma(self):
        """Set gamma"""

    @abstractmethod
    def get_image(self):
        """Get image from camera"""