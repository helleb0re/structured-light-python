from __future__ import annotations

import os
import json
from datetime import datetime
from typing import List, Tuple

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
from fpp_structures import FPPMeasurement, PhaseShiftingAlgorithm, CameraMeasurement


from examples.test_plate_phasogrammetry import process_with_phasogrammetry

def initialize_cameras(
    camera_type: str, 
    projector: Projector=None, 
    cam_to_found_number: int = 2, 
    cameras_serial_numbers: List[str] = []
    ) -> list[Camera]:
    '''
    Search for connected cameras of specified type and links them with projector instance, returns list of detected cameras

    Args:
        camera_type (str): type of cameras to search
        projector (Projector): porjector instance to link with cameras instancies
        cam_to_found_number (int): number of cameras to search
        cameras_serial_numbers (List[str]): list of cameras' serial numbers to search

    Returns:
        cameras (list[camera]): list of detected cameras
    '''
    if camera_type == 'web':
        cameras = CameraWeb.get_available_cameras(cam_to_found_number)
    elif camera_type == 'baumer':
        cameras = CameraBaumer.get_available_cameras(cam_to_found_number, cameras_serial_numbers)
    elif camera_type == 'simulated':
        cameras = CameraSimulated.get_available_cameras(cam_to_found_number)
        # Set projector for simulated cameras
        if projector is not None:
            for camera in cameras:
                camera.projector = projector
    return cameras


def adjust_cameras(cameras: list[Camera]) -> None:
    '''
    Adjust camera capture parameters (focus length, exposure time, etc)
    with visual control
    '''
    for i, camera in enumerate(cameras):
        if camera.type == "web":
            camera_adjust(camera)
        elif camera.type == "baumer":
            exposure, gamma, gain = camera_baumer_adjust(camera)
            config.CAMERA_EXPOSURE[i] = exposure
            config.CAMERA_GAIN[i] = gain
            config.CAMERA_GAMMA[i] = gamma
    # Save calibration data to file
    config.save_calibration_data()


def calibrate_projector(cameras: list[Camera], projector: Projector) -> None:
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
    lam = lambda x, a, b, c: a * (x + c) ** b

    # Fit gamma function parameters for reduced brightness vs intensity sequence
    popt, pcov = optimize.curve_fit(lam, int_reduced, brt_reduced, p0=(1, 1, 1))
    print(
        f"Fitted gamma function - Iout = {popt[0]:.3f} * (Iin + {popt[2]:.3f}) ^ {popt[1]:.3f}"
    )

    # Draw fitted gamma function
    gg = lam(intensity, *popt)

    plt.plot(intensity, brightness, "b+")
    plt.plot(intensity, gg, "r-")
    plt.xlabel("Intensity, relative units")
    plt.ylabel("Brightness, relative units")
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
    brt_corrected, _ = get_brightness_vs_intensity(
        cameras, projector, use_correction=True
    )

    # Draw corrected brightness vs intensity
    plt.plot(intensity, brt_corrected, "b+")
    plt.xlabel("Intensity, relative units")
    plt.ylabel("Brightness, relative units")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


def get_brightness_vs_intensity(cameras : List[Camera], projector: Projector, use_correction: bool) -> Tuple(List[float], List[float]):
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
    max_intensity = 1024
    average_num = 5
    border_width = 20

    projector.set_up_window()

    # TODO: Make generic to number of cameras
    brightness1 = []
    brightness2 = []

    # Make thin black and white borders
    image = np.zeros((projector.height, projector.width))
    image[border_width:-border_width, border_width:-border_width] = max_intensity

    temp_img = cameras[0].get_image()

    for intensity in range(max_intensity):
        image[2 * border_width: -2 * border_width, 2 * border_width: -2 * border_width] = intensity / max_intensity
        projector.project_pattern(image, use_correction)

        img1 = np.zeros(temp_img.shape, dtype=np.float64)
        img2 = np.zeros(temp_img.shape, dtype=np.float64)

        for _ in range(average_num):
            cv2.waitKey(config.MEASUREMENT_CAPTURE_DELAY)

            img1 = img1 + cameras[0].get_image()
            img2 = img2 + cameras[1].get_image()
        
        img1 = img1 / average_num
        img2 = img2 / average_num
        roi_x = slice(int(img1.shape[1] / 2 - win_size_x), int(img1.shape[1] / 2 + win_size_x))
        roi_y = slice(int(img1.shape[0] / 2 - win_size_y), int(img1.shape[0] / 2 + win_size_y))
        brt1 = np.mean(img1[roi_y, roi_x]) / max_intensity
        brt2 = np.mean(img2[roi_y, roi_x]) / max_intensity

        brightness1.append(brt1)
        brightness2.append(brt2)

        img_to_display1 = img1.astype(np.uint16)
        cv2.rectangle(
            img_to_display1,
            (roi_x.start, roi_y.start),
            (roi_x.stop, roi_y.stop),
            (255, 0, 0), 3,
        )
        cv2.putText(
            img_to_display1,
            f"{intensity = }",
            (50, 50),
            cv2.FONT_HERSHEY_PLAIN,
            5, (255, 0, 0), 2,
        )
        cv2.putText(
            img_to_display1,
            f"Brightness = {brt1:.3f}",
            (50, 100),
            cv2.FONT_HERSHEY_PLAIN,
            5, (255, 0, 0), 2,
        )
        cv2.imshow('cam1', img_to_display1)

        img_to_display2 = img2.astype(np.uint16)
        cv2.rectangle(
            img_to_display2,
            (roi_x.start, roi_y.start),
            (roi_x.stop, roi_y.stop),
            (255, 0, 0), 3,
        )
        cv2.putText(
            img_to_display2,
            f"{intensity = }",
            (50, 50),
            cv2.FONT_HERSHEY_PLAIN,
            5, (255, 0, 0), 2,
        )
        cv2.putText(
            img_to_display2,
            f"Brightness = {brt2:.3f}",
            (50, 100),
            cv2.FONT_HERSHEY_PLAIN,
            5, (255, 0, 0), 2,
        )
        cv2.imshow('cam2', img_to_display2)

    projector.close_window()
    cv2.destroyWindow('cam1')
    cv2.destroyWindow('cam2')

    return brightness1, brightness2


def capture_measurement_images(
    cameras: List[Camera],
    projector: Projector, 
    phase_shift_type: PhaseShiftingAlgorithm = PhaseShiftingAlgorithm.n_step
    ) -> FPPMeasurement:
    '''
    Do fringe projection measurement. Generate pattern, project them via projector and capture images with cameras.

    Args:
        cameras (list[Camera]): list of available cameras to capture measurement images
        projector (Projector): porjector to project patterns
        vertical (bool): create vertical patterns, if False create horizontal

    Returns:
        meas (FPPMeasurement): measurement for first and second camera
    '''
    # Create OpenCV GUI windows to show captured images
    cv2.namedWindow('cam1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam1', 600, 400)
    cv2.namedWindow('cam2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam2', 600, 400)

    shift_num = 4
    frequencies = [1, 4, 12, 48, 90]

    # Create phase shift profilometry patterns
    patterns_v, _ = create_psp_templates(
        config.PROJECTOR_WIDTH,
        config.PROJECTOR_HEIGHT,
        frequencies,
        phase_shift_type,
        shifts_number=shift_num,
        vertical=True,
    )
    patterns_h, phase_shifts = create_psp_templates(
        config.PROJECTOR_WIDTH,
        config.PROJECTOR_HEIGHT,
        frequencies,
        phase_shift_type,
        shifts_number=shift_num,
        vertical=False,
    )

    patterns_vh = {'vertical': patterns_v, 'horizontal': patterns_h}

    cam_results = [
        CameraMeasurement(fringe_orientation='vertical'),
        CameraMeasurement(fringe_orientation='vertical'),
        CameraMeasurement(fringe_orientation='horizontal'),
        CameraMeasurement(fringe_orientation='horizontal'),
    ]

    # Create FPPMeasurement instance with results
    meas = FPPMeasurement(phase_shift_type, frequencies, phase_shifts, cam_results)

    # Create folders to save measurement results if defined in config
    if config.SAVE_MEASUREMENT_IMAGE_FILES:
        measure_name = f'{datetime.now():%d-%m-%Y_%H-%M-%S}'
        last_measurement_path = f'{config.DATA_PATH}/{measure_name}'
        os.makedirs(f'{last_measurement_path}/')
        os.makedirs(f'{last_measurement_path}/{config.CAMERAS_FOLDER_NAMES[0]}/')
        os.makedirs(f'{last_measurement_path}/{config.CAMERAS_FOLDER_NAMES[1]}/')

    # Set up projector
    projector.set_up_window()

    for res1, res2 in ((cam_results[0], cam_results[1]), (cam_results[2], cam_results[3])):
        
        orientation = res1.fringe_orientation
        patterns = patterns_vh[orientation]

        # Iter thru generated patterns
        for i in range(len(patterns)):

            if config.SAVE_MEASUREMENT_IMAGE_FILES:
                res1.imgs_file_names.append([])
                res2.imgs_file_names.append([])
            else:
                res1.imgs_list.append([])
                res2.imgs_list.append([])

            for j in range(len(patterns[i])):
                projector.project_pattern(patterns[i][j])

                # Capture one frame before measurement for wecams
                if cameras[0].type == 'web':
                    cameras[0].get_image()
                if cameras[1].type == 'web':
                    cameras[1].get_image()

                # Wait delay time before pattern projected and images captures
                cv2.waitKey(config.MEASUREMENT_CAPTURE_DELAY)

                # Capture images
                frames_1 = []
                frames_2 = []
                for _ in range(1):
                    frames_1.append(cameras[0].get_image())
                    frames_2.append(cameras[1].get_image())

                frame_1 = np.mean(frames_1, axis=0).astype(np.uint8)
                frame_2 = np.mean(frames_2, axis=0).astype(np.uint8)

                cv2.imshow('cam1', frame_1)
                cv2.imshow('cam2', frame_2)

                # Save images if defined in config
                if config.SAVE_MEASUREMENT_IMAGE_FILES:
                    filename1 = f'{last_measurement_path}/{config.CAMERAS_FOLDER_NAMES[0]}/' + config.IMAGES_FILENAME_MASK.format(i, j, orientation)
                    filename2 = f'{last_measurement_path}/{config.CAMERAS_FOLDER_NAMES[1]}/' + config.IMAGES_FILENAME_MASK.format(i, j, orientation)
                    saved1 = cv2.imwrite(filename1, frame_1)
                    saved2 = cv2.imwrite(filename2, frame_2)

                    # Store saved images filenames
                    if saved1 and saved2:
                        res1.imgs_file_names[-1].append(filename1)
                        res2.imgs_file_names[-1].append(filename2)
                    else:
                        raise Exception('Error during image saving!')
                else:
                    res1.imgs_list[-1].append(frame_1)
                    res2.imgs_list[-1].append(frame_2)

    # Stop projector
    projector.close_window()

    # Close OpenCV GUI windows
    cv2.destroyWindow('cam1')
    cv2.destroyWindow('cam2')

    # Save results of measurement in json file if defined in config
    if config.SAVE_MEASUREMENT_IMAGE_FILES:        
        with open(f'{last_measurement_path}/' + config.MEASUREMENT_FILENAME_MASK.format(measure_name), 'x') as f:
            json.dump(meas, f, ensure_ascii=False, indent=4, default=vars)
        config.LAST_MEASUREMENT_PATH = last_measurement_path
        config.save_calibration_data()

    return meas


if __name__ == '__main__':

    projector = Projector(
        config.PROJECTOR_WIDTH,
        config.PROJECTOR_HEIGHT,
        config.PROJECTOR_MIN_BRIGHTNESS,
        config.PROJECTOR_MAX_BRIGHTNESS,
    )

    cameras = initialize_cameras(config.CAMERA_TYPE, projector, cam_to_found_number=2)

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
            if int(answer) not in choices:
                raise Exception()
        except:
            continue
        else:
            choice = int(answer)

        if choice == 0:
            break

        elif choice == 1:
            adjust_cameras(cameras)

        elif choice == 2:
            calibrate_projector(cameras, projector)

        elif choice == 3:
            frequencies = [1, 4, 16, 64, 100, 120]
            test_pattern, _ = create_psp_templates(
                config.PROJECTOR_WIDTH,
                config.PROJECTOR_HEIGHT,
                frequencies,
                PhaseShiftingAlgorithm.n_step,
                1,
                vertical=False,
            )
            MinMaxProjectorCalibration(test_pattern, cameras, projector)

        elif choice == 4:
            measurement = capture_measurement_images(
                cameras, projector, phase_shift_type=PhaseShiftingAlgorithm.n_step
            )
            process_with_phasogrammetry(measurement)
