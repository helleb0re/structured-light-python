import cv2
import json


def camera_adjust(camera):

    def on_focus_change(value):
        camera.focus = value

    def on_exposure_change(value):
        camera.exposure = value * -0.1
    
    def on_brightness_change(value):
        camera.brightness = value
    
    def on_gamma_change(value):
        camera.gamma = (value - 50) * 0.1

    cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam', 800, 600)

    cv2.createTrackbar('Focus', 'cam', 0, 100, on_focus_change)
    cv2.createTrackbar('Exposure', 'cam', 0, 100, on_exposure_change)
    cv2.createTrackbar('Brightness', 'cam', 0, 200, on_brightness_change)
    cv2.createTrackbar('Gamma', 'cam', 0, 100, on_gamma_change)

    while (True):
        img = camera.get_image()
        cv2.imshow('cam', img)
        k = cv2.waitKey(1)
        if k == 27:
            break

    focus = cv2.getTrackbarPos('Focus', 'cam')
    exposure = cv2.getTrackbarPos('Focus', 'cam') * -0.1
    brightness = cv2.getTrackbarPos('Brightness', 'cam')
    gamma = (cv2.getTrackbarPos('Gamma', 'cam') - 50) * 0.1

    cv2.destroyAllWindows()
    return focus, exposure, brightness, gamma


def camera_baumer_adjust(camera):

    def on_exposure_change(value):
        camera.exposure = 5000 + 150 * value

    def on_gamma_change(value):
        camera.gamma = 1 + (value - 50) * 0.01

    def on_gain_change(value):
        camera.gain = 1 + 0.02 * value

    cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam', 800, 600)

    cv2.createTrackbar('Exposure', 'cam', 0, 200, on_exposure_change)
    cv2.createTrackbar('Gain', 'cam', 0, 100, on_gain_change)
    cv2.createTrackbar('Gamma', 'cam', 0, 100, on_gamma_change)

    while True:
        img = camera.get_image()
        cv2.imshow('cam', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    
    # cv2.destroyAllWindows()

    exposure = 5000 + 150 * cv2.getTrackbarPos('Exposure', 'cam')
    gamma = 1 + (cv2.getTrackbarPos('Gamma', 'cam') - 50) * 0.01
    gain = 1 + 0.02 * cv2.getTrackbarPos('Gain', 'cam')

    cv2.destroyAllWindows()
    return exposure, gamma, gain