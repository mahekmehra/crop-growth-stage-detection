import cv2
import numpy as np


# =========================
# BRIGHTNESS CONTROL
# =========================

def brighten_image(image):

    return cv2.convertScaleAbs(
        image,
        alpha=1.4,
        beta=40
    )


def darken_image(image):

    return cv2.convertScaleAbs(
        image,
        alpha=0.45,
        beta=-50
    )


# =========================
# GAUSSIAN NOISE
# =========================

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


# =========================
# SALT & PEPPER NOISE
# =========================

def add_salt_pepper_noise(image):

    noisy = np.copy(image)

    prob = 0.02

    black = 0
    white = 255

    probs = np.random.random(image.shape[:2])

    noisy[probs < (prob / 2)] = black

    noisy[probs > 1 - (prob / 2)] = white

    return noisy


# =========================
# MOTION BLUR
# =========================

def add_motion_blur(image):

    kernel_size = 15

    kernel = np.zeros(
        (kernel_size, kernel_size)
    )

    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

    kernel = kernel / kernel_size

    blurred = cv2.filter2D(
        image,
        -1,
        kernel
    )

    return blurred


# =========================
# LOW LIGHT SIMULATION
# =========================

def low_light_simulation(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hsv[:,:,2] = hsv[:,:,2] * 0.35

    low_light = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    return low_light


# =========================
# REMOVE GAUSSIAN NOISE
# =========================

def remove_gaussian_noise(image):

    return cv2.GaussianBlur(
        image,
        (5,5),
        0
    )


# =========================
# NON LOCAL MEANS DENOISING
# =========================

def advanced_denoising(image):

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21
    )


# =========================
# SHARPEN IMAGE
# =========================

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


# =========================
# CLAHE ENHANCEMENT
# =========================

def clahe(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe_obj = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8,8)
    )

    cl = clahe_obj.apply(l)

    merged = cv2.merge((cl,a,b))

    enhanced = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# =========================
# GAMMA CORRECTION
# =========================

def gamma_correction(image, gamma=1.5):

    invGamma = 1.0 / gamma

    table = np.array([

        ((i / 255.0) ** invGamma) * 255

        for i in np.arange(0,256)

    ]).astype("uint8")

    return cv2.LUT(image, table)


# =========================
# FILTERS
# =========================

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