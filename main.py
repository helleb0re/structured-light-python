import os
import json
from datetime import datetime

import cv2
import neoapi
import numpy as np
from matplotlib import pyplot as plt

from ROI import ROI
from camera_baumer import CameraBaumer

from camera_web import CameraWeb
from projector import Projector
from create_patterns import create_psp_template, create_psp_templates
from hand_set_up_camera import camera_adjust, camera_baumer_adjust
from calibration_patterns import calibration_patterns
from data_experiment import ExperimentSettings
from fpp_structures import FPPMeasurement


def detect_cameras(cam_type='web', amount=2):
    '''
    Detect connected cameras
    '''
    if cam_type == 'web':
        return [CameraWeb(number=i) for i in range(amount)]
    elif cam_type == 'baumer':
        return [CameraBaumer(neoapi.Cam()) for i in range(amount)]


def adjust_cameras(cameras):
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


def capture_measurement_images(cameras, projector, patterns):

    cv2.namedWindow('cam1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam1', 600, 400)
    cv2.namedWindow('cam2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam2', 600, 400)
  
    patterns, phase_shifts, frequencies = create_psp_templates(1920, 1080, 7)

    filenames1 = []
    filenames2 = []

    measure_name = f'{datetime.now():%d-%m-%Y_%H-%M-%S}'
    os.makedirs(f'./data/{measure_name}/')
    os.makedirs(f'./data/{measure_name}/cam_1/')
    os.makedirs(f'./data/{measure_name}/cam_2/')

    for i, pattern in enumerate(patterns):
        for j, _ in enumerate(projector.project_patterns(pattern)):

            if cameras[0].type == 'web':
                _1 = cameras[0].get_image()
            if cameras[1].type == 'web':
                _2 = cameras[1].get_image()

            while True:
                frame_1 = cameras[0].get_image()
                frame_2 = cameras[1].get_image()
                if cameras[0].type == 'web':
                    frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
                if cameras[1].type == 'web':
                    frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
                k = cv2.waitKey(1)
                cv2.imshow('cam1', frame_1)
                cv2.imshow('cam2', frame_2)
                if k == 27:
                    filename1 = f'./data/{measure_name}/cam_1/frame_{i}_{j}.png'
                    filename2 = f'./data/{measure_name}/cam_2/frame_{i}_{j}.png'
                    saved1 = cv2.imwrite(filename1, frame_1)
                    saved2 = cv2.imwrite(filename2, frame_2)

                    if saved1 and saved2:
                        filenames1.append(filename1)
                        filenames2.append(filename2)
                    else:
                        raise Exception('Error during image saving!')
                    break

    cv2.destroyWindow('cam1')
    cv2.destroyWindow('cam2')

    meas1 = FPPMeasurement(
        frequencies, phase_shifts, filenames1
    )
    meas2 = FPPMeasurement(
        frequencies, phase_shifts, filenames2
    )

    with open(f"./data/{measure_name}/measure_{measure_name}.json", "x") as f:
                json.dump((meas1, meas2), f, ensure_ascii=False, indent=4, default=vars)

    return meas1, meas2

def define_ROI(cameras, projector):
    cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam', 600, 400)

    # background photos
    background_photos = []
    for cam in cameras:
        gray = cv2.cvtColor(cam.get_image(), cv2.COLOR_BGR2GRAY)
        background_photos.append(gray)
    
    # define measuring area
    photos_with_patterns = []
    test_pattern = create_psp_template(1920, 1080, 200, 1)
    for _ in projector.project_patterns(test_pattern):
        for cam in cameras:
            cv2.waitKey()
            __ = cam.get_image()
            gray = cv2.cvtColor(cam.get_image(), cv2.COLOR_BGR2GRAY)
            cv2.imshow("cam", gray)
            cv2.waitKey()
            photos_with_patterns.append(gray)

    # measuring_area = [[] for _ in range(len(cameras))]
    # for i in range(len(patterns2)):
    #     for j in range(len(cameras)):
    #         # binarization mb
    #         measuring_area[j].append(high_fringe_photos[j][i] - background_photos[j])
    #         cv2.imwrite('./data/res/cam_{}/{}_{}.png'.format(j+1, 'measuring_area', i), measuring_area[j][i])
    for i, (bg_photo, fringe_photo) in enumerate(zip(background_photos, test_pattern)):
        tmp = fringe_photo - bg_photo
        cv2.waitKey()
        # тут пока лишь ручное добавление
        if (i == 0):
            roi = ROI(1920, 1080)
            roi.corners[0].x = 442
            roi.corners[1].x = 1347
            roi.corners[2].x = 1365
            roi.corners[3].x = 436

            roi.corners[0].x = 263
            roi.corners[1].x = 97
            roi.corners[2].x = 785
            roi.corners[3].x = 764
            ExperimentSettings.add_ROI_values(roi)
        elif (i == 1):
            roi = ROI(1920, 1080)
            roi.corners[0].x = 497
            roi.corners[1].x = 1505
            roi.corners[2].x = 1545
            roi.corners[3].x = 514

            roi.corners[0].x = 145
            roi.corners[1].x = 185
            roi.corners[2].x = 726
            roi.corners[3].x = 787
            ExperimentSettings.add_ROI_values(roi)
    
        cv2.destroyWindow('cam')

if __name__ == '__main__':
    
    patterns = []

    with open("config.json") as f:
        data = json.load(f)
    projector = Projector(
        int(data["projector"]["width"]),
        int(data["projector"]["height"]),
        int(data["projector"]["min_brightness"]),
        int(data["projector"]["max_brightness"]))

    cameras = detect_cameras(cam_type='baumer', amount=2)

    choices = {i for i in range(5)}

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
            test_pattern, _ = create_psp_template(1920, 1080, 10, 1)
            calibration_patterns(test_pattern, cameras, projector)

        elif (choice == 4):
            meas1, meas2 = capture_measurement_images(cameras, projector, patterns)
        # except:
        #     print("Введите одно из представленных значений на выбор.")
        #     continue
