import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt


def show_results(img1, img2, global_title='Test', text1='Original', text2='Modificada'):
    plt.figure(figsize=(12, 6))
    plt.title(global_title)

    # Imagen original
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(text1)
    plt.axis('off')

    # Imagen modificada
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(text2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()



def get_pupil(in_image):
    height, width = np.shape(in_image)
    blur = cv.medianBlur(in_image, 5)
    pupils = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, width / 2, param1=120, param2=30, minRadius=height // 14,
                              maxRadius=height // 7)
    return pupils


def get_iris(in_image):
    height, width = np.shape(in_image)
    blur = cv.medianBlur(in_image, 5)
    iris = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, width / 1.1, param1=100, param2=2.8, minRadius=height // 8,
                              maxRadius=height * 2 // 8)
    return iris


def get_images(directory: str):
    imgs = os.listdir(directory)
    images = []
    for i in range(len(imgs)):
        images.append(np.uint8(cv.imread(directory + '/' + imgs[i], cv.IMREAD_GRAYSCALE)))
    return images

def detect_eye():
    # Initialize image, pupils and iris lists
    image_list = get_images('test_images')
    pupils = []
    irises = []
    for image in image_list:
        # Convert to color for representation
        image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        # Draw the pupil and iris
        pupil = get_pupil(image)
        iris = get_iris(image)

        if pupil is not None:
            pupil = np.uint16(np.around(pupil))
            for i in pupil[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                # pupil center
                cv.circle(image_color, center, 1, (0, 100, 100), 3)
                # pupil outline
                cv.circle(image_color, center, radius, (255, 0, 255), 2)
        if iris is not None:
            iris = np.uint16(np.around(iris))
            for i in iris[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                # iris center
                cv.circle(image_color, center, 1, (0,100,100), 3)
                # iris outline
                cv.circle(image_color, center, radius, (0,200,100), 2)

        cv.imshow("detected pupils and iris", image_color)
        cv.waitKey(0)

detect_eye()