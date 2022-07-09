import os
import json
from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt

from camera_baumer import CameraBaumer

from camera import Camera
from camera_web import CameraWeb
from projector import Projector
from create_patterns import create_psp_template, create_psp_templates
from hand_set_up_camera import camera_adjust, camera_baumer_adjust
from calibration_patterns import calibration_patterns
from data_experiment import ExperimentSettings
from fpp_structures import FPPMeasurement
from processing import calculate_phase_for_fppmeasurement


def initialize_cameras(cam_type: str='web', cam_to_found_number: int=2) -> list[Camera]:
    '''
    Detect connected cameras
    '''
    if cam_type == 'web':
        cameras = CameraWeb.get_available_cameras(cam_to_found_number)
    elif cam_type == 'baumer':
        cameras = CameraBaumer.get_available_cameras(cam_to_found_number)
        
    return cameras


def adjust_cameras(cameras: list[Camera]) -> None:
    '''
    Adjust camera capture parameters (focus length, exposure time, etc)
    with visual control
    '''
    with open("config.json") as f:
        data = json.load(f)

    for i, camera in enumerate(cameras):
        if camera.type == 'web':
            camera_adjust(camera)
        elif camera.type == 'baumer':
            exposure, gamma, gain = camera_baumer_adjust(camera)
            data["cameras"]["baumer"][i]["exposure"] = exposure
            data["cameras"]["baumer"][i]["gamma"] = gamma
            data["cameras"]["baumer"][i]["gain"] = gain

    with open("config.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def calibrate_projector(cameras : list[Camera], projector: Projector) -> None :
    '''
    Сalibrate projector image with gamma correction
    '''
    win_size_x = 50
    win_size_y = 50

    projector.set_up_window()
    
    brightness_vs_intensity = []

    image = np.zeros((projector.height, projector.width))
    
    for intensity in range(256):
        image[:,:] = intensity
        projector.project_pattern(image)

        img1 = cameras[0].get_image()
        # img2 = cameras[1].get_image()

        brightness = np.mean(img1[img1.shape[0]/2 - win_size_x:img1.shape[0]/2 + win_size_x, img1.shape[1]/2 - win_size_y:img1.shape[1]/2 + win_size_y])

        brightness_vs_intensity.append(brightness)

    projector.close_window()

    plt.plot(brightness_vs_intensity)
    plt.show()

    # TODO: calculate gamma coeficient and store it in Projector instance


def capture_measurement_images(cameras: list[Camera], projector: Projector, time_delay: float, save_meas_params: bool = False) -> tuple[FPPMeasurement, FPPMeasurement]:

    cv2.namedWindow('cam1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam1', 600, 400)
    cv2.namedWindow('cam2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam2', 600, 400)

    patterns, phase_shifts, frequencies = create_psp_templates(1920, 1080, 7)

    images1 = []
    images2 = []

    if save_meas_params:
        filenames1 = []
        filenames2 = []
        measure_name = f'{datetime.now():%d-%m-%Y_%H-%M-%S}'
        os.makedirs(f'./data/{measure_name}/')
        os.makedirs(f'./data/{measure_name}/cam_1/')
        os.makedirs(f'./data/{measure_name}/cam_2/')

    projector.set_up_window()

    for i in range(len(patterns)):

        if save_meas_params:
            fn_for_one_freq1 = []
            fn_for_one_freq2 = []

        img_for_one_freq1 = []
        img_for_one_freq2 = []

        for j in range(len(patterns[i])):
            projector.project_pattern(patterns[i][j])

            if cameras[0].type == 'web':
                _1 = cameras[0].get_image()
            if cameras[1].type == 'web':
                _2 = cameras[1].get_image()

            while True:
                cv2.waitKey(time_delay)

                frame_1 = cameras[0].get_image()
                frame_2 = cameras[1].get_image()

                if save_meas_params:
                    filename1 = f'./data/{measure_name}/cam_1/frame_{i}_{j}.png'
                    filename2 = f'./data/{measure_name}/cam_2/frame_{i}_{j}.png'
                    saved1 = cv2.imwrite(filename1, frame_1)
                    saved2 = cv2.imwrite(filename2, frame_2)

                    if saved1 and saved2:
                        fn_for_one_freq1.append(filename1)
                        fn_for_one_freq2.append(filename2)

                img_for_one_freq1.append(frame_1)
                img_for_one_freq2.append(frame_2)
                    # else:
                    #     raise Exception('Error during image saving!')
                break

            if save_meas_params:
                filenames1.append(fn_for_one_freq1)
                filenames2.append(fn_for_one_freq2)

        images1.append(img_for_one_freq1)
        images2.append(img_for_one_freq2)
    
    projector.close_window()

    cv2.destroyWindow('cam1')
    cv2.destroyWindow('cam2')

    meas1 = FPPMeasurement(
        frequencies, phase_shifts, filename1 if save_meas_params else [], images1
    )
    meas2 = FPPMeasurement(
        frequencies, phase_shifts, filename2 if save_meas_params else [], images2
    )

    if save_meas_params:
        with open(f"./data/{measure_name}/measure_{measure_name}.json", "x") as f:
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

    with open("config.json") as f:
        data = json.load(f)
    projector = Projector(
        int(data["projector"]["width"]),
        int(data["projector"]["height"]),
        int(data["projector"]["min_brightness"]),
        int(data["projector"]["max_brightness"]))

    cameras = initialize_cameras(cam_type='baumer', cam_to_found_number=2)

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
            time_delay = int(data['capture_parameters']['delay'])
            measurements = capture_measurement_images(cameras, projector, time_delay)
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
