from __future__ import annotations

import os
import json
from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt

import config
from camera import Camera
from projector import Projector
from camera_web import CameraWeb
from camera_baumer import CameraBaumer
from camera_simulated import CameraSimulated
from create_patterns import create_psp_template, create_psp_templates
from hand_set_up_camera import camera_adjust, camera_baumer_adjust
from calibration_patterns import calibration_patterns
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
    '''
    win_size_x = 50
    win_size_y = 50

    cv2.namedWindow('cam1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam1', 600, 400)
    # cv2.namedWindow('cam2', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('cam2', 600, 400)

    projector.set_up_window()
    
    brightness_vs_intensity = []

    image = np.zeros((projector.height, projector.width))
    
    k = 5
    temp_img = cameras[0].get_image()

    for intensity in range(256):
        print(f'Calibrate {intensity = }')
        image[:,:] = intensity
        projector.project_pattern(image, False)

        img1 = np.zeros(temp_img.shape, dtype=np.float64)
        
        for _ in range(k):
            cv2.waitKey(config.MEASUREMENT_CAPTURE_DELAY)

            img1 = img1 + cameras[0].get_image()
            # img2 = cameras[1].get_image()
        
        img1 = img1 / k
        roi_x = slice(int(img1.shape[1]/2 - win_size_x), int(img1.shape[1]/2 + win_size_x))
        roi_y = slice(int(img1.shape[0]/2 - win_size_y), int(img1.shape[0]/2 + win_size_y))
        brightness = np.mean(img1[roi_y, roi_x])

        brightness_vs_intensity.append(brightness)

        img_to_display = img1.astype(np.uint8)
        cv2.rectangle(img_to_display, (roi_x.start, roi_y.start), (roi_x.stop, roi_y.stop), (255, 0, 0), 3)
        cv2.imshow('cam1', img_to_display)

    projector.close_window()

    plt.plot(brightness_vs_intensity)
    plt.show()

    # TODO: calculate gamma coeficient and store it in Projector instance

    cv2.destroyWindow('cam1')
    # cv2.destroyWindow('cam2')

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

def define_ROI(cameras: list[Camera], projector: Projector) -> None:

    projector.set_up_window()
    
    white_screen = np.ones(projector.resolution)

    cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam', 600, 400)

    # background photos
    background_photos = []
    for cam in cameras:
        k = 0
        while k != 27:
            #__ = cam.get_image()
            gray = cam.get_image()
            if gray.shape[2] > 1:
                gray = cv2.cvtColor(cam.get_image(), cv2.COLOR_BGR2GRAY)
            cv2.imshow("cam", gray)
            k = cv2.waitKey(100)
        background_photos.append(gray)
    
    # define measuring area
    whiteround_photos = []

    projector.project_pattern(white_screen)

    for cam in cameras:
        k = 0
        while k != 27:
            #__ = cam.get_image()
            gray = cam.get_image()
            if gray.shape[2] > 1:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            cv2.imshow("cam", gray)
            k = cv2.waitKey(100)
        whiteround_photos.append(gray)

    projector.close_window()

    # i = 0
    # for photo1, photo2 in zip(background_photos, whiteround_photos):
    #     cv2.imwrite(f'./data/background_{i}.png', photo1)
    #     cv2.imwrite(f'./data/whiteground_{i}.png', photo2)
    #     i = i + 1
    
    cv2.destroyWindow('cam')


if __name__ == '__main__':

    projector = Projector(
        config.PROJECTOR_WIDTH,
        config.PROJECTOR_HEIGHT,
        config.PROJECTOR_MIN_BRIGHTNESS,
        config.PROJECTOR_MAX_BRIGHTNESS)

    cameras = initialize_cameras(cam_type=config.CAMERA_TYPE, cam_to_found_number=2)

    choices = {i for i in range(6)}

    while True:
        print(f"Подключено {len(cameras)} камер")
        print("==========================================================")
        print("1 - Настройка параметров камеры")
        print("2 - Определение области проецированния шаблонов")
        print("3 - Калибровка проецируемого изображения")
        print("4 - Проведение измерения")
        print("==========================================================")
        print("0 - Выход из программы")
        answer = input("Введите что-нибудь из предложенного списка выше: ")

        # try:
        #     if int(answer) not in choices: raise Exception()
        #     else:
        choice = int(answer)

        if (choice == 0):
            break

        elif (choice == 1):
            adjust_cameras(cameras)

        elif (choice == 2):
            define_ROI(cameras, projector)

        elif (choice == 3):
            # test_pattern, _, _ = create_psp_templates(1920, 1080, 7, 1)
            # calibration_patterns(test_pattern, cameras, projector)
            calibrate_projector(cameras, projector)

        elif (choice == 4):
            measurements = capture_measurement_images(cameras, projector)
            w_phases_1, uw_phases_1 = calculate_phase_for_fppmeasurement(measurements[0])
            w_phases_2, uw_phases_2 = calculate_phase_for_fppmeasurement(measurements[1])

            for w_phase_1, uw_phase_1, w_phase_2, uw_phase_2 in zip(
                w_phases_1, uw_phases_1, w_phases_2, uw_phases_2):
                plt.subplot(221)
                plt.imshow(w_phase_1, cmap="gray")
                plt.colorbar()

                plt.subplot(222)
                plt.imshow(uw_phase_1, cmap="gray")
                plt.colorbar()

                plt.subplot(223)
                plt.imshow(w_phase_2, cmap="gray")
                plt.colorbar()

                plt.subplot(224)
                plt.imshow(uw_phase_2, cmap="gray")
                plt.colorbar()

                plt.show()

        # except:
        #     print("Введите одно из представленных значений на выбор.")
        #     continue
