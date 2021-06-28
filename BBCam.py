import sys
import getopt
import math
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2 as opencv
import numpy as np
import pyvirtualcam as vcam

# many of the methods thanks to the very useful helper functions in:
# https://github.com/akinuri/dump/blob/master/python/opencv/feathered-edges/helpers_cv2.py

def threshold(img, thresh=128, maxval=255, type=opencv.THRESH_BINARY):
    if len(img.shape) == 3:
        img = opencv.cvtColor(img, opencv.COLOR_BGR2GRAY)
    threshed = opencv.threshold(img, thresh, maxval, type)[1]
    return threshed


def dilate_mask(mask, kernel_size=11):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = opencv.dilate(mask, kernel, iterations=1)
    return dilated


def smooth_mask(mask, kernel_size=11):
    blurred = opencv.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    if len(blurred.shape) == 3:
        blurred = opencv.cvtColor(blurred, opencv.COLOR_BGR2GRAY)
    threshed = opencv.threshold(blurred, 128, 255, opencv.THRESH_BINARY)[1]
    return threshed


def alpha_blend(background, foreground, mask):
    mask = mask.astype("float") / 255.
    foreground = foreground.astype("float") / 255.
    background = background.astype("float") / 255.
    out = background * (1 - mask) + foreground * mask
    out = (out * 255).astype("uint8")
    return out


def odd(num):
    if isinstance(num, float):
        num = math.floor(num)
    if num % 2 == 0:
        num = num - 1
    return num


if __name__ == "__main__":
    # tensorflow.js model will be converted by tfjs_graph_converter
    bodypix_model = load_model(download_model(BodyPixModelPaths.RESNET50_FLOAT_STRIDE_32))

    # get video capture device (webcam)
    cap = opencv.VideoCapture(0)
    success, frame = cap.read()

    if not success:
        print("Error getting a webcam image!")
        sys.exit(1)

    # params
    cam_fps = 20
    show_cam = bool(True)
    debug_cam = bool(False)
    mask_threshold_val = 0.75
    mask_dilation_val = 51
    mask_smooth_val = odd(mask_dilation_val * 1.5)
    mask_blur_val = 51
    bg_blur_val = 31

    # get commandline args
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "", ["headless", "debug", "fps=","mtresh=", "mdil=", "msmooth=", "mblur=", "bblur="])

    for opt, arg in opts:
        if opt in '--fps':
            cam_fps = int(arg)
        elif opt in '--mtresh':
                mask_threshold_val = min(max(0.0, float(arg)), 1.0)
        elif opt in '--mdil':
            mask_dilation_val = int(arg)
        elif opt in '--msmooth':
            mask_smooth_val = int(arg)
        elif opt in '--mblur':
            mask_blur_val = int(arg)
        elif opt in '--bblur':
            bg_blur_val = int(arg)
        elif opt in '--headless':
            show_cam = bool(False)
        elif opt in '--debug':
            debug_cam = bool(True)

    # create camera
    with vcam.Camera(width=frame.shape[1], height=frame.shape[0], fps=cam_fps, fmt=vcam.PixelFormat.BGR) as cam:
        print(f'Using virtual camera: {cam.device} {cam.width}x{cam.height} @ {cam_fps} fps')
        print('params:')
        print(f'mask threshold value= {mask_threshold_val}')
        print(f'mask dilation value= {mask_dilation_val}')
        print(f'mask smooth value= {mask_smooth_val}')
        print(f'mask blur value= {mask_blur_val}')
        print(f'background blur value= {bg_blur_val}')
        print()

        # iterate frames
        while cap.isOpened():
            success, frame = cap.read()

            # get bodypix prediction mask
            result = bodypix_model.predict_single(frame)
            mask = result.get_mask(threshold=mask_threshold_val).numpy().astype(np.uint8)

            # mask is normalised by default
            mask = np.multiply(mask, 255)

            if debug_cam:
                opencv.imshow('frame', frame)
                opencv.imshow('mask', mask)

            # dilate the mask
            mask_dilated = dilate_mask(mask, mask_dilation_val)

            if debug_cam:
                opencv.imshow('mask_dilated', mask_dilated)

            # smoothen out the mask
            mask_smooth = smooth_mask(mask_dilated, mask_smooth_val)

            if debug_cam:
                opencv.imshow('mask_smooth', mask_smooth)

            # blur the mask
            mask_blurred = opencv.GaussianBlur(mask_smooth, (mask_blur_val, mask_blur_val), 0)
            mask_blurred = opencv.cvtColor(mask_blurred, opencv.COLOR_GRAY2BGR)

            if debug_cam:
                opencv.imshow('mask_blurred', mask_blurred)

            # thresh the mask (back to [0-1] range)
            mask_threshed = threshold(mask_blurred, 1)

            if debug_cam:
                opencv.imshow('mask_threshed', mask_threshed)

            # blur the frame to use as background
            bg = opencv.blur(frame, (bg_blur_val, bg_blur_val), 0)

            # blend the blurred with the non-blurred frame using the mask
            output = alpha_blend(bg, frame, mask_blurred)

            # show frame in window
            if show_cam:
                opencv.imshow('BBcam', output)

            cam.send(output)
            cam.sleep_until_next_frame()

            if opencv.waitKey(1) == 27:
                break  # esc to quit

    # release video capture device
    cap.release()

    # close imshow frame
    opencv.destroyAllWindows()
