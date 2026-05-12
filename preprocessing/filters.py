import cv2
import numpy as np


def brighten_image(image):

    return cv2.convertScaleAbs(
        image,
        alpha=1.3,
        beta=30
    )


def darken_image(image):

    return cv2.convertScaleAbs(
        image,
        alpha=0.5,
        beta=-40
    )


def add_noise(image):

    noise = np.random.normal(
        0,
        25,
        image.shape
    )

    noisy = image + noise

    noisy = np.clip(
        noisy,
        0,
        255
    )

    return noisy.astype(np.uint8)

def remove_gaussian_noise(image):

    return cv2.GaussianBlur(
        image,
        (5,5),
        0
    )


def gaussian_filter(image):

    return cv2.GaussianBlur(
        image,
        (5,5),
        0
    )


def median_filter(image):

    return cv2.medianBlur(
        image,
        5
    )


def bilateral_filter(image):

    return cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )


def sharpen_image(image):

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


def clahe(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe_obj = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    return clahe_obj.apply(gray)