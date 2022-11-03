'''Module with helper functions to work with Baumer cameras'''

import os
import sys
import json
import datetime
import time
import glob
from pathlib import Path
import multiprocessing as mp

from functools import reduce

import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np

import cv2

import neoapi
#import PYbgapi2

from camera_web import CameraWeb
from camera_baumer import CameraBaumer

"""
def _get_image(camera_id):
    '''Get image from PYbgapi2 camera.

    Arguments:
    camera_id -- id of camera to capture image from
    '''
    im = PYbgapi2.get_image_from_stream(camera_id)
    width = im['width']
    height = im['height']
    buffer = im['image_array']
    frameID = im['frameID']
    timestamp = im['timestamp']
    pixel_format = im['pixel_format']

    if (width > 0 and height > 0):
        print(f'Grab image {width}x{height} succefully from camera {camera_id} -- frameID={frameID}, timestamp={timestamp}')
        #buf = np.frombuffer(buffer, dtype=np.uint8, count=height*width).reshape((height, width))   
        #return buf
        if pixel_format == 'Mono8':
            format = 'L'
        else:
            format = 'I;16'

        return Image.frombytes(format, (width, height), buffer, 'raw') 
    else:
        print(f'Error while grabbing image from camera {camera_id}')
        print(PYbgapi2.get_log())
        return None 
"""

def calibrate_cameras(markers_x, markers_y, images_count=15, use_stream=True,
                      wait_period=3000, save_calibrating_images=True,
                      save_path='', blur_threshold=100):
    '''Capture images for calibrating stereosystem with chessboard pattern. Function capture images
    from two Baumer cameras and wait for a chessboard appears on both images. If all points of pattern
    are detected by cv2.findChessboardCorners on both images these images are saved as files.

    Keyword arguments:
    images_count -- count of images to use for calibration (default 15)
    use_stream -- use stream option of PYbgapi2 (default True)
    wait_period -- period of time between two images captured for calibration (default 1000 ms)
    save_calibrating_images -- save images which are used for calibration (default True)
    save_path -- path to save images
    '''

    '''
    print(f'Init PYbgapi2 system -- {PYbgapi2.init_system()}')
    cameras = Camera.get_camera_list(PYbgapi2.get_camera_names())

    if len(cameras) == 0:
        print(f'Cameras in PYbgapi2 system not found')
        return

    print(f'Cameras in PYbgapi2 system -- {cameras}')

    while not reduce(lambda x,y: x.started and y.started, cameras):
        print(f'Cameras not started, try to reinit system...')
        PYbgapi2.deinit_system()
        PYbgapi2.init_system()
        cameras = Camera.get_camera_list(PYbgapi2.get_camera_names())
    '''

    print(f'Init Baumer NeoAPI system ...')
    
    cameras = CameraBaumer.get_available_cameras(cameras_num_to_find=2)

    for cam_num, camera in enumerate(cameras):
        cv2.namedWindow(f"camera_{cam_num}", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(f"camera_{cam_num}", 800, 600)

    images = [[] for _ in cameras]

    left_upper_corners = [[], []]
    right_bottom_corners = [[], []]

    calibrate_time = time.time()

    i = 0

    main_cornes_founded = False

    while True:
        # Get images and measure capturing time
        start_time = time.time()

        for cam_num, camera in enumerate(cameras):
            images[cam_num] = camera.get_image()

        end_time = time.time() - start_time                                    
        
        if images[0] is not None and images[1] is not None:          
            # Bool to store if corners is found on images or not
            # cornes_founded = True
            cornes_founded = [False, False]
            # Variables to store corners for area 
            lu_corner = [None, None]
            rb_corner = [None, None]

            # Wait for wait_period to move chessboard on images
            do_calibrate = (time.time() - calibrate_time) > wait_period / 1000

            for k, img in enumerate(images):
                gray = img.copy()
                if cameras[k].type == "web":
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img_to_draw = gray.copy()
                img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_GRAY2RGB)

                blur_index = cv2.Laplacian(gray, cv2.CV_64F).var()

                # Try to find chessboard on images       
                if do_calibrate and blur_index > blur_threshold:
                    # termination criteria
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                    objp = np.zeros((markers_y*markers_x,3), np.float32)
                    objp[:,:2] = np.mgrid[0:markers_x,0:markers_y].T.reshape(-1,2)

                    # Find the chess board corners
                    ret, corners = cv2.findChessboardCorners(gray, (markers_x, markers_y), flags=cv2.CALIB_CB_FAST_CHECK)

                    # If found, add object points, image points (after refining them)
                    if ret == True:
                        cornes_founded[k] = True
                        # Store left upper and right bottom corners
                        lu_corner[k] = corners[0]
                        rb_corner[k] = corners[-1]

                        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)                            

                        calibrate_time = time.time()

                        # Draw and display the corners
                        img_to_draw = cv2.drawChessboardCorners(img_to_draw, (markers_x, markers_y), corners2, ret)                                                

                # Draw calibrated area on image
                fill_area = np.array(img_to_draw)
                if left_upper_corners[0] and left_upper_corners[1]:
                    for j, _ in enumerate(left_upper_corners[k]):
                        fill_area = cv2.rectangle(fill_area,
                                                    (int(left_upper_corners[k][j][0][0]), int(left_upper_corners[k][j][0][1])),
                                                    (int(right_bottom_corners[k][j][0][0]), int(right_bottom_corners[k][j][0][1])),
                                                    (255, 0, 0), -1)
                    img_to_draw = cv2.addWeighted(img_to_draw, 0.7, fill_area, 0.3, 0)

                
                cv2.putText(img_to_draw, f'Images captured {i} from {images_count}', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0), 2)
                cv2.putText(img_to_draw, f'Blur index {blur_index:.2f}', (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0), 2)
                cv2.imshow(f"camera_{k}", img_to_draw)
                k = cv2.waitKey(10)
                if k == 27: # Escape
                    return
                elif k != -1:
                    print(k)
            
            if all(cornes_founded) and main_cornes_founded:
                for k, _ in enumerate(images):
                    left_upper_corners[k].append(lu_corner[k])
                    right_bottom_corners[k].append(rb_corner[k])

                if save_calibrating_images:
                    for cam_num, image in enumerate(images):
                        cv2.imwrite(f'{save_path}camera_{cam_num}_image{i}.tif', image)

                if i == images_count:
                    break

                i = i + 1

                cornes_founded[k] = False
                main_cornes_founded = False
            elif cornes_founded and not main_cornes_founded:
                main_cornes_founded = True
        else:
            print('Failed to grab images from camera 0 and 1')


def calculate_calibration(force_recalculate=False, file_mask1='camera_2_image*.png', file_mask2='camera_1_image*.png', camera_type = "web"):
    markers_x = 25 #37
    markers_y = 17 #23

    square_size_x = 15 # mm
    square_size_y = 15 # mm

    # VCXG-32M
    sensor_x_size = 6.9632 # mm
    sensor_y_size = 5.2224 # mm

    # VLG-24M
    # sensor_x_size = 7.06 # mm
    # sensor_y_size = 5.29 # mm
 
    data_loaded = False

    if not force_recalculate:
        try:
            with open('calibrated_data.json', 'r') as fp:
                calibration_data = json.load(fp)
                data_loaded = True
                print('Calibration data is load from calibration_data.json') 
        except:
            print('Calibration data is not founded in calibration_data.json') 

    if not data_loaded or force_recalculate:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((markers_x*markers_y, 3), np.float32)
        objp[:,:2] = np.mgrid[0:markers_y, 0:markers_x].T.reshape(-1,2)

        objp[:,1] *= square_size_x
        objp[:,0] *= square_size_y

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space

        imgpoints1 = [] # 2d points in image plane.
        imgpoints2 = []

        images_for_camera1 = glob.glob(file_mask1)
        images_for_camera2 = glob.glob(file_mask2)

        cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img1', 800, 600)
        cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img2', 800, 600)

        files_to_delete = []

        for fname1, fname2 in zip(images_for_camera1, images_for_camera2):
            img1 = cv2.imread(fname1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(fname2, cv2.IMREAD_GRAYSCALE)
            gray1 = img1.copy()
            gray2 = img2.copy()

            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

            # if camera_type == "web":
            # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # cv2.imshow('img1', img1)
            # cv2.waitKey()

            # Find the chess board corners
            ret1, corners1 = cv2.findChessboardCorners(gray1, (markers_y, markers_x), cv2.CALIB_CB_ADAPTIVE_THRESH)
            ret2, corners2 = cv2.findChessboardCorners(gray2, (markers_y, markers_x), cv2.CALIB_CB_ADAPTIVE_THRESH)

            if corners1[0,0,0] + corners1[0,0,1] > corners1[-1,0,0] + corners1[-1,0,1]:
                corners1 = corners1[::-1,:,:].copy()
            if corners2[0,0,0] + corners2[0,0,1] > corners2[-1,0,0] + corners2[-1,0,1]:
                corners2 = corners2[::-1,:,:].copy()

            # If found, add object points, image points (after refining them)
            if ret1 and ret2:
                objpoints.append(objp)
                corners_subpix1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners_subpix2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                imgpoints1.append(corners_subpix1)
                imgpoints2.append(corners_subpix2)

                # Draw and display the corners
                cv2.drawChessboardCorners(img1, (markers_y, markers_x), corners_subpix1, ret1)
                cv2.drawChessboardCorners(img2, (markers_y, markers_x), corners_subpix2, ret2)
                cv2.waitKey(100)
            else:
                files_to_delete.append((fname1, fname2))

            cv2.imshow('img1', img1)
            cv2.imshow('img2', img2)
            cv2.waitKey(100)

        for file in files_to_delete:
            images_for_camera1.remove(file[0])
            images_for_camera2.remove(file[1])

        cv2.destroyAllWindows()

        camera_matrix = np.array([[50, 0, gray1.shape[1]/2],[0, 50, gray1.shape[0]/2],[0,0,1]])
        dist_coef = np.zeros(12)

        ret1, mtx1, dist1, rvecs1, tvecs1, stdDeviationsIntrinsics1, stdDeviationsExtrinsics1, perViewErrors1 = cv2.calibrateCameraExtended(objpoints, imgpoints1, gray1.shape[::-1], camera_matrix, dist_coef, flags=cv2.CALIB_FIX_PRINCIPAL_POINT, criteria=criteria)
        fovx1, fovy1, focalLength1, principalPoint1, aspectRatio1 = cv2.calibrationMatrixValues(mtx1, gray1.shape[::-1], sensor_x_size, sensor_y_size)

        print('Camera 1 calibration results:')
        print(f'RMS error {ret1:<15.4f}')
        print(f'Camera matrix:')
        print(f'{mtx1}')
        print(f'Focal length {focalLength1:<15.2f}')

        ret2, mtx2, dist2, rvecs2, tvecs2, stdDeviationsIntrinsics2, stdDeviationsExtrinsics2, perViewErrors2 = cv2.calibrateCameraExtended(objpoints, imgpoints2, gray1.shape[::-1], camera_matrix, dist_coef, flags=cv2.CALIB_FIX_PRINCIPAL_POINT, criteria=criteria)
        fovx2, fovy2, focalLength2, principalPoint2, aspectRatio2 = cv2.calibrationMatrixValues(mtx2, gray1.shape[::-1], sensor_x_size, sensor_y_size)

        print('Camera 2 calibration results:')
        print(f'RMS error {ret2:<15.4f}')
        print(f'Camera matrix:')
        print(f'{mtx2}')
        print(f'Focal length {focalLength2:<15.2f}')

        retval = 10
        perViewErrors = None

        while retval > 1 or np.max(perViewErrors) > 1:            
            if perViewErrors is not None:
                std = np.std(perViewErrors)
                avg = np.average(perViewErrors)
                
                outliers = []
                for i in range(len(perViewErrors)):
                    if np.average(perViewErrors[i]) > 3*std and np.average(perViewErrors[i]) > 1.2*avg:
                        outliers.append(i)
                if len(outliers) == 0:
                    print(f'Stereo calibrate stoped at retval = {retval:.3f} with {len(perViewErrors)} images in calibration set as no outliers is founded...')
                    break
                for i in sorted(outliers, reverse=True):
                    objpoints.pop(i)
                    imgpoints1.pop(i)
                    imgpoints2.pop(i)
                print(f'Stereo calibrate itteration, outliers {len(outliers)} founded...  retval = {retval:.3f}')
            retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, perViewErrors = cv2.stereoCalibrateExtended(objpoints, imgpoints1, imgpoints2,
                                                                                                                                    mtx1, dist1, mtx2, dist2,
                                                                                                                                    gray1.shape[::-1], None, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 +
                                                                                                                                    + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_SAME_FOCAL_LENGTH)
        else:
            print(f'Stereo calibrate stoped at retval = {retval:.3f} with {len(perViewErrors)} images in calibration set')
 
        _, _, focalLength1, _, _ = cv2.calibrationMatrixValues(cameraMatrix1, gray1.shape[::-1], sensor_x_size, sensor_y_size)

        print('Camera 1 after stereocalibration results:')
        print(f'RMS error {retval:<15.4f}')
        print(f'Camera matrix:')
        print(f'{cameraMatrix1}')
        print(f'Focal length {focalLength1:<15.2f}')

        _, _, focalLength2, _, _ = cv2.calibrationMatrixValues(cameraMatrix2, gray2.shape[::-1], sensor_x_size, sensor_y_size)

        print('Camera 2 after stereocalibration results:')
        print(f'RMS error {retval:<15.4f}')
        print(f'Camera matrix:')
        print(f'{cameraMatrix2}')
        print(f'Focal length {focalLength2:<15.2f}')

        print(f'Distance between cameras {np.sum(T**2)**0.5:<15.2f}')


        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, gray1.shape[::-1], R, T, alpha=-1, flags=0)

        mapx1, mapy1 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray1.shape[::-1], cv2.CV_32F)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray1.shape[::-1], cv2.CV_32F)

        width = max(roi1[2], roi2[2])
        height = max(roi1[3], roi2[3])
 
        for fname1, fname2 in zip(images_for_camera1, images_for_camera2):
            img1 = cv2.imread(fname1)
            img2 = cv2.imread(fname2)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            img_rect1 = cv2.remap(gray1, mapx1, mapy1, cv2.INTER_LINEAR)[roi1[1]:roi1[1]+height, roi1[0]:roi1[0]+width]
            img_rect2 = cv2.remap(gray2, mapx2, mapy2, cv2.INTER_LINEAR)[roi2[1]:roi2[1]+height, roi2[0]:roi2[0]+width]
            
            # draw the images side by side
            total_size = (max(img_rect1.shape[0], img_rect2.shape[0]), img_rect1.shape[1] + img_rect2.shape[1])
            img = np.zeros(total_size, dtype=np.uint8)
            img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
            img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2
            
            # draw horizontal lines every 25 px accross the side by side image
            for i in range(20, img.shape[0], 25):
                cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

            cv2.namedWindow('imgRectified', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('imgRectified', 550, 450)
            cv2.imshow('imgRectified', img)            
            cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('img1', 550, 450)
            cv2.imshow('img1', cv2.remap(gray1, mapx1, mapy1, cv2.INTER_LINEAR))
            cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('img2', 550, 450)
            cv2.imshow('img2', cv2.remap(gray2, mapx2, mapy2, cv2.INTER_LINEAR))
            cv2.waitKey(100)
        

        calibration_data = {
            'camera_0':
                {'camera_id': 0,
                'ret': ret1,
                'mtx': mtx1.tolist(),
                'dist': dist1.tolist(), 
                'rvecs': [el.tolist() for el in rvecs1],
                'tvecs': [el.tolist() for el in tvecs1],
                'perViewErrors': perViewErrors1.tolist()
                },
            'camera_1':
                {'camera_id': 1,
                'ret': ret2,
                'mtx': mtx2.tolist(),
                'dist': dist2.tolist(), 
                'rvecs': [el.tolist() for el in rvecs2],
                'tvecs': [el.tolist() for el in tvecs2],
                'perViewErrors': perViewErrors2.tolist()
                },
            'R': R.tolist(),
            'T': T.tolist(),
            'ret': retval,
            'perViewErrors': perViewErrors.tolist()
            }
        
        with open('calibrated_data.json', 'w') as fp:
            json.dump(calibration_data, fp, indent=4)
            print('Calibration data is saved to calibration_data.json') 

        return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T

    else:
        return np.array(calibration_data['camera_0']['mtx']), np.array(calibration_data['camera_0']['dist']), \
               np.array(calibration_data['camera_1']['mtx']), np.array(calibration_data['camera_1']['dist']), \
               np.array(calibration_data['R']), np.array(calibration_data['T'])


def experiment_registration():

    path_to_save = os.path.join('C:/exp_img/', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    background_substruct = False
    preprocessing = False
    threshold = 70
    registration = False
    reg_image_num = 0
    capturing = True

    '''
    print(f'Init PYbgapi2 system -- {PYbgapi2.init_system()}')
    cameras = Camera.get_camera_list(PYbgapi2.get_camera_names())

    if len(cameras) == 0:
        print(f'Cameras in PYbgapi2 system not found')
        return

    print(f'Cameras in PYbgapi2 system -- {cameras}')

    while not reduce(lambda x,y: x.started and y.started, cameras):
        print(f'Cameras not started, try to reinit system...')
        PYbgapi2.deinit_system()
        PYbgapi2.init_system()
        cameras = Camera.get_camera_list(PYbgapi2.get_camera_names())
    '''

    print(f'Init Baumer NeoAPI system ...')
    cameras = [Camera(), Camera()]

    for camera in cameras:
        camera.gain = 5
        camera.triger_mode = neoapi.TriggerMode_On
        camera.pixel_format = neoapi.PixelFormat_Mono8

    for camera in cameras:
        cv2.namedWindow(camera.name, cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(camera.name, 550, 450)

    images = [[] for camera in cameras]

    while capturing:
        # Get images and measure capturing time

        start_time = time.time()

        for i, camera in enumerate(cameras):
            images[i] = camera.get_image()

        end_time = time.time() - start_time                                    
            
        print(f'Images from cameras grabbed in {end_time} sec')

        for i, im in enumerate(images):   
            if im is not None:

                im = np.array(im)

                if background_substruct:
                    im = cv2.subtract(im, background[i])

                if preprocessing:
                    if im.dtype == np.uint16:
                        max_value = 4095
                    else:
                        max_value = 255
                    _, im = cv2.threshold(im, threshold, max_value, cv2.THRESH_BINARY)
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
                    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

                if im.dtype == np.uint16:
                    im = cv2.normalize(im, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
                cv2.imshow(cameras[i].name, im)

                if registration:
                    if os.path.exists(path_to_save):
                        cv2.imwrite(os.path.join(path_to_save, f'camera{i}/IMG{reg_image_num:>04}.tif'), im)
                    else:
                        os.makedirs(path_to_save)
                        for cam_num, _ in enumerate(cameras):
                            os.makedirs(os.path.join(path_to_save, f'camera{cam_num}'))

        if registration:
            reg_image_num += 1   
        
        key = cv2.waitKey(1)
        if key == 27:    # Esc key to stop
            capturing = False
        elif key == -1:  # normally -1 returned,so don't print it
            continue
        elif key == 32:  # Space
            images[0].save('test0.tiff')
            images[1].save('test1.tiff')
        elif key == 104:    # h
            if registration:
                registration = False
                print(f'Image registration is OFF')
            else:
                registration = True
                reg_image_num = 0
                print(f'Image registration is ON')
        elif key == 119:     # w
            for camera in cameras:
                camera.gain += 1
                print(f'Camera {camera.name} gain set to {camera.gain}')
        elif key == 115:    # s
            for camera in cameras:
                camera.gain -= 1
                print(f'Camera {camera.name} gain set to {camera.gain}')
        elif key == 100:    # d
            for camera in cameras:
                camera.exposure_time += 1000
                print(f'Camera {camera.name} exposure time set to {camera.exposure_time} us')
        elif key == 97:    # a
            for camera in cameras:
                camera.exposure_time -= 1000
                print(f'Camera {camera.name} exposure time set to {camera.exposure_time} us')
        elif key == 113:     # q
            if cameras[0].triger_mode.value == neoapi.TriggerMode_On:
                value = neoapi.TriggerMode_Off
            else:
                value = neoapi.TriggerMode_On
            for camera in cameras:
                camera.triger_mode = value
                print(f'Camera {camera.name} trigger mode set to {camera.triger_mode.GetString()}')
        elif key == 101:    # e
            if cameras[0].pixel_format.value ==  neoapi.PixelFormat_Mono8:
                value = neoapi.PixelFormat_Mono12
            else:
                value = neoapi.PixelFormat_Mono8
            for camera in cameras:
                camera.pixel_format = value
                print(f'Camera {camera.name} pixel format set to {camera.pixel_format.GetString()}')
        elif key == 98:    # b
            if not background_substruct:
                background = [[] for camera in cameras]
                for i in range(len(cameras)):
                    background[i] = np.array(images[i]) 
                print(f'Backgrounds saved. Background substruct is on')
                background_substruct = True
            else:
                background_substruct = False
                print(f'Background substruct is off')
        elif key == 112:    # p
            preprocessing = not preprocessing
            print(f'Preprocessing is -- {preprocessing}')
        elif key == 49: # 1
            if threshold > 0:
                threshold -= 1
                print(f'Threshold = {threshold}')
        elif key == 50: # 2
            if threshold < 255:
                threshold += 1
                print(f'Threshold = {threshold}')
        else:
            print(key) # else print its value 


    #for i in range(len(cameras)):
        #print(f'Stop camera {i} in PYbgapi2 system -- {PYbgapi2.stop_camera(i)}')

    #print(f'DeInit PYbgapi2 system -- {PYbgapi2.deinit_system()}')


if __name__ == '__main__':

    MARKERS_X = 25 #37 #17
    MARKERS_Y = 17 #23 #13

    CALIBRATE_IMAGES_COUNT = 50
    CALIBRATE_IMAGES_PATH = r'.\calibrate_images\\'
    CALIBRATE_FILE_MASK_1 = r'.\calibrate_images\camera_0_image*.tif'
    CALIBRATE_FILE_MASK_2 = r'.\calibrate_images\camera_1_image*.tif'
    RECALCULATE_CALIBRATION = True

    # calibrate_cameras(MARKERS_X, MARKERS_Y, images_count=CALIBRATE_IMAGES_COUNT, save_calibrating_images=True, save_path=CALIBRATE_IMAGES_PATH, blur_threshold=50)

    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T = calculate_calibration(RECALCULATE_CALIBRATION, CALIBRATE_FILE_MASK_1, CALIBRATE_FILE_MASK_2, camera_type='baumer')

    #experiment_registration()