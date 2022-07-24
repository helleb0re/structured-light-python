from __future__ import annotations

import os
import json
from datetime import datetime

import cv2
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

import config
from camera import Camera
from projector import Projector
from camera_web import CameraWeb
from camera_baumer import CameraBaumer
from camera_simulated import CameraSimulated
from create_patterns import create_psp_templates
from hand_set_up_camera import camera_adjust, camera_baumer_adjust
from min_max_projector_calibration import MinMaxProjectorCalibration
from fpp_structures import FPPMeasurement
from processing import calculate_phase_for_fppmeasurement


def initialize_cameras(cam_type: str, cam_to_found_number: int=2) -> list[Camera]:
    '''
    Detect connected cameras
    '''
    if cam_type == 'web':
        cameras = CameraWeb.get_available_cameras(cam_to_found_number)
    elif cam_type == 'baumer':
        cameras = CameraBaumer.get_available_cameras(cam_to_found_number)
    elif cam_type == 'simulated':
        cameras = CameraSimulated.get_available_cameras(cam_to_found_number)
        # Set projector for simulated cameras
        for camera in cameras:
            camera.projector = projector
    return cameras


def adjust_cameras(cameras: list[Camera]) -> None:
    '''
    Adjust camera capture parameters (focus length, exposure time, etc)
    with visual control
    '''
    for i, camera in enumerate(cameras):
        if camera.type == 'web':
            camera_adjust(camera)
        elif camera.type == 'baumer':
            exposure, gamma, gain = camera_baumer_adjust(camera)
            config.CAMERA_EXPOSURE[i] = exposure
            config.CAMERA_GAIN[i] = gain
            config.CAMERA_GAMMA[i] = gamma
    # Save calibration data to file
    config.save_calibration_data()


def calibrate_projector(cameras : list[Camera], projector: Projector) -> None :
    '''
    Сalibrate projector image with gamma correction

    Args:
        cameras (list[Camera]): list of available cameras to capture measurement images
        projector (Projector): porjector to project patterns
    '''
    brightness, _ = get_brightness_vs_intensity(cameras, projector, use_correction=False)

    # Calculate gamma coeficient
    # Mare intensity linsapce
    intensity = np.linspace(0, np.max(brightness), len(brightness))

    # Find saturation level
    saturation_level = 0.95
    k = 0
    for i in range(len(intensity)):
        if brightness[i] > np.max(brightness) * saturation_level:
            k = k + 1
            if k > 3:
                saturation = i - 2
                break

    # Reduce sequency to saturation level
    int_reduced = intensity[:saturation]
    brt_reduced = brightness[:saturation]

    # Gamma function to fit
    lam = lambda x,a,b,c: a*(x + c)**b

    # Fit gamma function parameters for reduced brightness vs intensity sequence
    popt, pcov = optimize.curve_fit(lam, int_reduced, brt_reduced, p0=(1,1,1))
    print(f'Fitted gamma function - Iout = {popt[0]:.3f} * (Iin + {popt[2]:.3f}) ^ {popt[1]:.3f}')

    # Draw fitted gamma function
    gg = lam(intensity, *popt)

    plt.plot(intensity, brightness, 'b+')
    plt.plot(intensity, gg, 'r-')
    plt.xlabel('Intensity, relative units')
    plt.ylabel('Brightness, relative units')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()

    # Store new gamma correction coefficients
    config.PROJECTOR_GAMMA_A = popt[0]
    config.PROJECTOR_GAMMA_B = popt[1]
    config.PROJECTOR_GAMMA_C = popt[2]
    config.save_calibration_data()

    # Check gamma correction
    brt_corrected, _ = get_brightness_vs_intensity(cameras, projector, use_correction=True)

    # Draw corrected brightness vs intensity
    plt.plot(intensity, brt_corrected, 'b+')
    plt.xlabel('Intensity, relative units')
    plt.ylabel('Brightness, relative units')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


def get_brightness_vs_intensity(cameras : list[Camera], projector: Projector, use_correction: bool) -> list[float]:
    '''
    Get brightness vs intensity dependence by projecting constant intensity
    on screen and capture images with cameras. Brightness is averaged in small
    region for several captured images.

    Args:
        cameras (list[Camera]): list of available cameras to capture measurement images
        projector (Projector): porjector to project patterns
        use_correction (bool): use correction to project patterns
    '''
    cv2.namedWindow('cam1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam1', 600, 400)
    cv2.namedWindow('cam2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam2', 600, 400)

    # TODO: Add parameters to config
    win_size_x = 50
    win_size_y = 50
    max_intensity = 256
    average_num = 5
    border_width = 20
    
    projector.set_up_window()
    
    # TODO: Make generic to number of cameras
    brightness1 = []
    brightness2 = []

    # Make thin black and white borders
    image = np.zeros((projector.height, projector.width))
    image[border_width:-border_width,border_width:-border_width] = max_intensity
    
    temp_img = cameras[0].get_image()

    for intensity in range(max_intensity):
        image[2*border_width:-2*border_width,2*border_width:-2*border_width] = intensity / max_intensity
        projector.project_pattern(image, use_correction)

        img1 = np.zeros(temp_img.shape, dtype=np.float64)
        img2 = np.zeros(temp_img.shape, dtype=np.float64)
        
        for _ in range(average_num):
            cv2.waitKey(config.MEASUREMENT_CAPTURE_DELAY)

            img1 = img1 + cameras[0].get_image()
            img2 = img2 + cameras[1].get_image()
        
        img1 = img1 / average_num 
        img2 = img2 / average_num 
        roi_x = slice(int(img1.shape[1]/2 - win_size_x), int(img1.shape[1]/2 + win_size_x))
        roi_y = slice(int(img1.shape[0]/2 - win_size_y), int(img1.shape[0]/2 + win_size_y))
        brt1 = np.mean(img1[roi_y, roi_x]) / max_intensity
        brt2 = np.mean(img2[roi_y, roi_x]) / max_intensity

        brightness1.append(brt1)
        brightness2.append(brt2)

        img_to_display1 = img1.astype(np.uint8)
        cv2.rectangle(img_to_display1, (roi_x.start, roi_y.start), (roi_x.stop, roi_y.stop), (255, 0, 0), 3)
        cv2.putText(img_to_display1, f'{intensity = }', (50,50), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 2)
        cv2.putText(img_to_display1, f'Brightness = {brt1:.3f}', (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 2)
        cv2.imshow('cam1', img_to_display1)

        img_to_display2 = img2.astype(np.uint8)
        cv2.rectangle(img_to_display2, (roi_x.start, roi_y.start), (roi_x.stop, roi_y.stop), (255, 0, 0), 3)
        cv2.putText(img_to_display2, f'{intensity = }', (50,50), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 2)
        cv2.putText(img_to_display2, f'Brightness = {brt2:.3f}', (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 2)
        cv2.imshow('cam2', img_to_display2)

    projector.close_window()
    cv2.destroyWindow('cam1')
    cv2.destroyWindow('cam2')

    return brightness1, brightness2


def capture_measurement_images(cameras: list[Camera], projector: Projector) -> tuple[FPPMeasurement, FPPMeasurement]:
    '''
    Do fringe projection measurement. Generate pattern, project them via projector and capture images with cameras.

    Args:
        cameras (list[Camera]): list of available cameras to capture measurement images
        projector (Projector): porjector to project patterns

    Returns:
        meas1 (FPPMeasurement): measurement for first camera
        meas2 (FPPMeasurement): measurement for second camera
    '''
    # Create OpenCV GUI windows to show captured images
    cv2.namedWindow('cam1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam1', 600, 400)
    cv2.namedWindow('cam2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam2', 600, 400)

    # Create phase shift profilometry patterns
    patterns, phase_shifts, frequencies = create_psp_templates(1920, 1080, [1, 4, 12, 48, 90])

    # List to store captured images
    images1 = []
    images2 = []

    # Create folders to save measurement results if defined in config
    if config.SAVE_MEASUREMENT_IMAGE_FILES:
        filenames1 = []
        filenames2 = []
        measure_name = f'{datetime.now():%d-%m-%Y_%H-%M-%S}'
        os.makedirs(f'{config.DATA_PATH}/{measure_name}/')
        os.makedirs(f'{config.DATA_PATH}/{measure_name}/{config.CAMERAS_FOLDER_NAMES[0]}/')
        os.makedirs(f'{config.DATA_PATH}/{measure_name}/{config.CAMERAS_FOLDER_NAMES[1]}/')

    # Set up projector
    projector.set_up_window()

    # Iter thru generated patterns
    for i in range(len(patterns)):

        if config.SAVE_MEASUREMENT_IMAGE_FILES:
            fn_for_one_freq1 = []
            fn_for_one_freq2 = []

        img_for_one_freq1 = []
        img_for_one_freq2 = []

        # Iter thru pattern with phase shifts for one frequency
        for j in range(len(patterns[i])):
            # Project pattern
            projector.project_pattern(patterns[i][j])

            # Capture one frame before measurement for wecams
            if cameras[0].type == 'web':
                _1 = cameras[0].get_image()
            if cameras[1].type == 'web':
                _2 = cameras[1].get_image()

            # Wait delay time before pattern projected and images captures
            cv2.waitKey(config.MEASUREMENT_CAPTURE_DELAY)

            # Capture images
            frame_1 = cameras[0].get_image()
            frame_2 = cameras[1].get_image()

            cv2.imshow('cam1', frame_1)
            cv2.imshow('cam2', frame_2)

            # Save images if defined in config
            if config.SAVE_MEASUREMENT_IMAGE_FILES:
                filename1 = f'{config.DATA_PATH}/{measure_name}/{config.CAMERAS_FOLDER_NAMES[0]}/' + config.IMAGES_FILENAME_MASK.format(i, j)
                filename2 = f'{config.DATA_PATH}/{measure_name}/{config.CAMERAS_FOLDER_NAMES[1]}/' + config.IMAGES_FILENAME_MASK.format(i, j)
                saved1 = cv2.imwrite(filename1, frame_1)
                saved2 = cv2.imwrite(filename2, frame_2)

                # Store saved images filenames
                if saved1 and saved2:
                    fn_for_one_freq1.append(filename1)
                    fn_for_one_freq2.append(filename2)
                else:
                    raise Exception('Error during image saving!')

            # Store images for one frequency in one list 
            img_for_one_freq1.append(frame_1)
            img_for_one_freq2.append(frame_2)

        # Store saved images filenames for one frequency
        if config.SAVE_MEASUREMENT_IMAGE_FILES:
            filenames1.append(fn_for_one_freq1)
            filenames2.append(fn_for_one_freq2)
        
        # Store list of images for one frequency in total set images list
        images1.append(img_for_one_freq1)
        images2.append(img_for_one_freq2)
    
    # Stop projector
    projector.close_window()

    # Close OpenCV GUI windows
    cv2.destroyWindow('cam1')
    cv2.destroyWindow('cam2')

    # Create FPPMeasurement instances with results
    meas1 = FPPMeasurement(
        frequencies,
        phase_shifts, 
        filenames1 if config.SAVE_MEASUREMENT_IMAGE_FILES else [],
        None if config.SAVE_MEASUREMENT_IMAGE_FILES else images1
    )
    meas2 = FPPMeasurement(
        frequencies,
        phase_shifts,
        filenames2 if config.SAVE_MEASUREMENT_IMAGE_FILES else [],
        None if config.SAVE_MEASUREMENT_IMAGE_FILES else images2
    )

    # Save results of measurement in json file if defined in config
    if config.SAVE_MEASUREMENT_IMAGE_FILES:
        with open(f'{config.DATA_PATH}/{measure_name}/' + config.MEASUREMENT_FILENAME_MASK.format(measure_name), 'x') as f:
            json.dump((meas1, meas2), f, ensure_ascii=False, indent=4, default=vars)

    return meas1, meas2


if __name__ == '__main__':

    projector = Projector(
        config.PROJECTOR_WIDTH,
        config.PROJECTOR_HEIGHT,
        config.PROJECTOR_MIN_BRIGHTNESS,
        config.PROJECTOR_MAX_BRIGHTNESS)

    cameras = initialize_cameras(cam_type=config.CAMERA_TYPE, cam_to_found_number=2)

    choices = {i for i in range(6)}

    while True:
        print(f"Connected {len(cameras)} camera(s)")
        print("==========================================================")
        print("1 - Adjust cameras")        
        print("2 - Projector gamma correction calibration")
        print("3 - Check brightness profile")
        print("4 - Take measurements")
        print("==========================================================")
        print("0 - Exit script")
        answer = input("Type something from the suggested list above: ")

        try:
            if int(answer) not in choices: raise Exception()
        except:
            continue
        else:
            choice = int(answer)

        if (choice == 0):
            break

        elif (choice == 1):
            adjust_cameras(cameras)

        elif (choice == 2):
            calibrate_projector(cameras, projector)

        elif (choice == 3):
            test_pattern, _, _ = create_psp_templates(1920, 1080, 7, 1)
            MinMaxProjectorCalibration(test_pattern, cameras, projector)

        elif (choice == 4):
            measurements = capture_measurement_images(cameras, projector)
            w_phases_1, uw_phases_1, _, _ = calculate_phase_for_fppmeasurement(measurements[0])
            w_phases_2, uw_phases_2, _, _ = calculate_phase_for_fppmeasurement(measurements[1])

            plt.subplot(121)
            plt.imshow(uw_phases_1[-1], cmap="gray")
            plt.colorbar()

            plt.subplot(122)
            plt.imshow(uw_phases_2[-1], cmap="gray")
            plt.colorbar()

            plt.show()
