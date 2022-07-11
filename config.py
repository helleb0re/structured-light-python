'''
Module to store program configuration

Calibration data is stored in a separate config.json file and loaded when the module is imported
'''

import json

# Projector configuration
# Projector resolution width and height in pixels
PROJECTOR_WIDTH = 1280
PROJECTOR_HEIGHT = 720

# Maximum and minimum brightness for projected image correction
# Given values are default, the current values are loaded from the calibration file
PROJECTOR_MIN_BRIGHTNESS = 0
PROJECTOR_MAX_BRIGHTNESS = 255

# Cameras configuration
# Number of cameras used in measurement
CAMERAS_COUNT = 2

# Type of cameras used in measurement
CAMERA_TYPE = 'baumer'

# Cameras parameters default values, the current values are loaded from the calibration file
CAMERA_EXPOSURE = (20000, 20000)
CAMERA_GAIN = (1, 1)
CAMERA_GAMMA = (1, 1)

# Measurement configuration
# Path to save measurement data
DATA_PATH = './data'

# Images filenames mask
IMAGES_FILENAME_MASK = 'frame_{0}_{1}.png'

# Measurement filenames mask
MEASUREMENT_FILENAME_MASK = 'measure_{0}.json'

# Cameras folders in measurment folder
CAMERAS_FOLDER_NAMES = ('cam1', 'cam2')

# Save measurement image files
SAVE_MEASUREMENT_IMAGE_FILES = False

# Delay between pattern projection and camera image capture in miliseconds
MEASUREMENT_CAPTURE_DELAY = 300 # ms

# File name for calibration data
CONFIG_FILENAME = r'./config.json'

# Load calibration data from json file
with open("config.json") as f:
    calibration_data = json.load(f)
    
    PROJECTOR_MIN_BRIGHTNESS = float(calibration_data["projector"]["min_brightness"])
    PROJECTOR_MAX_BRIGHTNESS = float(calibration_data["projector"]["max_brightness"])
    
    CAMERA_EXPOSURE = (int(calibration_data['cameras']['baumer'][0]['exposure']),
                       int(calibration_data['cameras']['baumer'][1]['exposure']))
    CAMERA_GAIN = (float(calibration_data['cameras']['baumer'][0]['gain']),
                   float(calibration_data['cameras']['baumer'][1]['gain']))
    CAMERA_GAMMA = (float(calibration_data['cameras']['baumer'][0]['gamma']),
                    float(calibration_data['cameras']['baumer'][1]['gamma']))

def save_calibration_data() -> None:
    '''
    Save calibration data to config.json file
    '''
    with open("config.json") as f:
        calibration_data = json.load(f)

    for i in range(CAMERAS_COUNT):
        calibration_data["cameras"]["baumer"][i]["exposure"] = CAMERA_EXPOSURE[i]
        calibration_data["cameras"]["baumer"][i]["gain"] = CAMERA_GAIN[i]
        calibration_data["cameras"]["baumer"][i]["gamma"] = CAMERA_GAIN[i]

    with open("config.json", "w") as f:
        json.dump(calibration_data, f, ensure_ascii=False, indent=4)