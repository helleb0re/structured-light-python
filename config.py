'''
Module to store program configuration

Calibration data is stored in a separate config.json file and loaded when the module is imported
'''

import json

# Projector configuration
# Projector resolution width and height in pixels
PROJECTOR_WIDTH = 1280
PROJECTOR_HEIGHT = 720
# OpenCV GUI windows shift relative to first screen of the system
PROJECTOR_WINDOW_SHIFT = 1920

# Maximum and minimum brightness for projected image correction
# Given values are default, the current values are loaded from the calibration file
PROJECTOR_MIN_BRIGHTNESS = 0.0
PROJECTOR_MAX_BRIGHTNESS = 1.0

# Gamma correction coefficients for formula Iout = a * (Iin + c) ^ b
# Given values are default, the current values are loaded from the calibration file
PROJECTOR_GAMMA_A = 1.0
PROJECTOR_GAMMA_B = 2.2
PROJECTOR_GAMMA_C = 0

# Cameras configuration
# Number of cameras used in measurement
CAMERAS_COUNT = 2

# Type of cameras used in measurement
CAMERA_TYPE = 'baumer'

# Cameras parameters default values, the current values are loaded from the calibration file
CAMERA_EXPOSURE = [20000, 20000]
CAMERA_GAIN = [1, 1]
CAMERA_GAMMA = [1, 1]

# Measurement configuration
# Path to save measurement data
DATA_PATH = './data'

# Images filenames mask
IMAGES_FILENAME_MASK = 'frame_{2}_{0}_{1}.png'

# Measurement filenames mask
MEASUREMENT_FILENAME_MASK = 'fpp_measurement.json'

# Cameras folders in measurment folder
CAMERAS_FOLDER_NAMES = ['cam1', 'cam2']

# Save measurement image files
SAVE_MEASUREMENT_IMAGE_FILES = False

# Delay between pattern projection and camera image capture in miliseconds
MEASUREMENT_CAPTURE_DELAY = 300 # ms

# File name for calibration data
CONFIG_FILENAME = r'./config.json'

# Use multiprocessing to increase speed of processing
USE_MULTIPROCESSING = False

# Number of Pools to use in parallel processing
POOLS_NUMBER = 5

# Path to last measurement results
LAST_MEASUREMENT_PATH = None


# Load calibration data from json file
try:
    with open('config.json') as f:
        calibration_data = json.load(f)
        
        try:
            PROJECTOR_MIN_BRIGHTNESS = float(calibration_data['projector']['min_brightness'])
            PROJECTOR_MAX_BRIGHTNESS = float(calibration_data['projector']['max_brightness'])

            PROJECTOR_GAMMA_A = float(calibration_data['projector']['gamma_a'])
            PROJECTOR_GAMMA_B = float(calibration_data['projector']['gamma_b'])
            PROJECTOR_GAMMA_C = float(calibration_data['projector']['gamma_c'])
        except:
            pass
        
        try:
            CAMERA_EXPOSURE = [int(calibration_data['cameras']['baumer'][0]['exposure']),
                               int(calibration_data['cameras']['baumer'][1]['exposure'])]
            CAMERA_GAIN = [float(calibration_data['cameras']['baumer'][0]['gain']),
                           float(calibration_data['cameras']['baumer'][1]['gain'])]
            CAMERA_GAMMA = [float(calibration_data['cameras']['baumer'][0]['gamma']),
                            float(calibration_data['cameras']['baumer'][1]['gamma'])]
        except:
            pass

        try:
            LAST_MEASUREMENT_PATH = calibration_data['measurements']['last_measurement_path']
        except:
            pass
except:
    pass


def save_calibration_data() -> None:
    '''
    Save calibration data to config.json file
    '''
    try:
        with open("config.json") as f:
            calibration_data = json.load(f)

            calibration_data['projector']['gamma_a'] = PROJECTOR_GAMMA_A
            calibration_data['projector']['gamma_b'] = PROJECTOR_GAMMA_B
            calibration_data['projector']['gamma_c'] = PROJECTOR_GAMMA_C

        for i in range(CAMERAS_COUNT):
            calibration_data["cameras"]["baumer"][i]["exposure"] = CAMERA_EXPOSURE[i]
            calibration_data["cameras"]["baumer"][i]["gain"] = CAMERA_GAIN[i]
            calibration_data["cameras"]["baumer"][i]["gamma"] = CAMERA_GAIN[i]

        calibration_data['measurements']['last_measurement_path'] = LAST_MEASUREMENT_PATH
    except:
        pass
    else:
        with open("config.json", "w") as f:
            json.dump(calibration_data, f, ensure_ascii=False, indent=4)