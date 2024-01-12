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


def fill_circle(image, circle, scale):
    out_image = np.zeros_like(image)

    center = (circle[0], circle[1])
    radius = round(circle[2] * scale)
    # fill
    cv.circle(out_image, center, radius, (255, 255, 255), -1)

    return out_image / 255


def get_artifacts(in_image, pupil, iris):
    # Get general objects
    canny = cv.Canny(in_image, 150, 200)

    binary_pupil_artifacts = fill_circle(in_image, pupil, 0.85) * canny
    binary_iris_artifacts = (fill_circle(in_image, iris, 0.85) - fill_circle(in_image, pupil, 1.25)) * canny

    return binary_pupil_artifacts, binary_iris_artifacts


def detect_eye(image):
    # Convert image to color for representation
    image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    # Get parameters for pupil and iris
    pupil = get_pupil(image)
    iris = get_iris(image)

    # Draw pupil's circle
    if pupil is not None:
        pupil = np.uint16(np.round(pupil))
        for i in pupil[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv.circle(image_color, center, radius, (255, 0, 255), 2)

    # Draw iris' circle
    if iris is not None:
        iris = np.uint16(np.round(iris))
        for i in iris[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv.circle(image_color, center, radius, (0, 200, 100), 2)

    # Get and draw the artifacts for pupil and iris (only if one of each detected previously)
    if pupil is not None and iris is not None and len(pupil[0]) == len(iris[0]) == 1:
        # Get binary masks for pupil and iris
        pupil_artifacts, iris_artifacts = get_artifacts(image, pupil[0][0], iris[0][0])

        # Get the coordinates where the masks have values
        pupil_artifacts_coordinates = np.nonzero(pupil_artifacts)
        iris_artifacts_coordinates = np.nonzero(iris_artifacts)

        # Draw the artifacts in the image
        image_color[pupil_artifacts_coordinates] = [0, 0, 255]      # Pupil's artifacts in red
        image_color[iris_artifacts_coordinates] = [255, 255, 0]     # Iris' artifacts in blue

    # Show image with segmentations
    cv.imshow("Detected segmentations", image_color)
    cv.waitKey(0)

detect_eye()