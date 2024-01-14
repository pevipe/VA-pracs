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


def get_pupil(in_image: np.ndarray) -> np.ndarray:
    height, width = np.shape(in_image)
    blur = cv.medianBlur(in_image, 5)
    pupils = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, width / 2, param1=120, param2=30, minRadius=height // 14,
                             maxRadius=height // 7)
    return pupils


def get_iris(in_image: np.ndarray) -> np.ndarray:
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


def fill_circle(image: np.ndarray, circle: np.ndarray, scale: float = 1.0) -> np.ndarray:
    out_image = np.zeros_like(image)

    center = (int(circle[0]), (circle[1]))
    radius = round(float(circle[2]) * scale)
    # fill
    cv.circle(out_image, center, radius, (255, 255, 255), -1)

    return out_image / 255


def get_artifacts(in_image: np.ndarray, pupil: np.ndarray, iris: np.ndarray):
    # Máscaras binarias de iris y pupila
    pupil_circle = fill_circle(in_image, pupil, 0.85)
    iris_circle = fill_circle(in_image, iris, 0.95) - fill_circle(in_image, pupil, 1.05)

    # Imágenes con píxeles de las zonas de pupila o iris
    image_pupil = (in_image * pupil_circle).astype(np.uint8)
    image_iris = (in_image * iris_circle).astype(np.uint8)

    # Índices de las zonas de iris y pupila
    indices_to_keep_pupil = np.nonzero(pupil_circle)
    indices_to_keep_iris = np.nonzero(iris_circle)

    # Artefactos pupila: threshold de 50 es suficiente
    _, art_pupila = cv.threshold(image_pupil, 50, 255, cv.THRESH_BINARY)
    art_pupila = art_pupila * pupil_circle

    # Para el iris:
    # 1. Ecualizo el histograma de las zonas de interés y hago la media de ese valor con el de ese píxel en la imagen
    #    original
    # 2. Hago el threshold sobre esos valores (adaptativa, en función de la media de la región de iris)
    # 3. Creo una matriz resultado con la imagen a 0
    # 4. Para los índices del iris, copio los valores del threshold

    # Primero, un pequeño suavizado
    blur = cv.blur(image_iris, (3, 3), 0)
    # Aplicar ecualización
    eq1 = cv.equalizeHist(blur[indices_to_keep_iris])
    for i in range(len(eq1)):
        # Hacer media ponderada entre la imagen ecualizada y la suavizada
        s = (eq1[i][0].astype(np.uint16) * 2 + blur[indices_to_keep_iris[0][i], indices_to_keep_iris[1][i]].astype(
            np.uint16))
        eq1[i][0] = s // 3
    mean_iris = np.mean(eq1)
    white_threshold = mean_iris + (255 - mean_iris) * .58
    black_threshold = mean_iris * .3

    # Aplicar thresholds superior e inferior (se consideran artefactos los que están por debajo del primero y por
    # encima del segundo). Después, sumar los arrays (disjuntos)
    _, art_iris1 = cv.threshold(eq1, black_threshold, 255, cv.THRESH_BINARY_INV)
    _, art_iris2 = cv.threshold(eq1, white_threshold, 255, cv.THRESH_BINARY)
    art_iris = art_iris1 + art_iris2

    # De volver a forma matricial los datos de threshold para los índices del iris
    art_iris_mat = np.zeros_like(in_image)
    for i in range(len(indices_to_keep_iris[0])):
        art_iris_mat[indices_to_keep_iris[0][i], indices_to_keep_iris[1][i]] = art_iris[i][0]

    return art_pupila, art_iris_mat


def segment_eye(image: np.ndarray):
    # Convert image to color for representation
    image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    # Get parameters for pupil and iris
    pupil = get_pupil(image)
    iris = get_iris(image)
    pupil_artifacts, iris_artifacts = None, None

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
        image_color[pupil_artifacts_coordinates] = [0, 0, 255]  # Pupil's artifacts in blue
        image_color[iris_artifacts_coordinates] = [255, 255, 0]  # Iris' artifacts in yellow

    # Generate Region Of Interest (para esclerótica)
    height, width = np.shape(image)
    min_y = max(iris[0][0][1] - iris[0][0][2] - pupil[0][0][2], 0)
    max_y = min(iris[0][0][1] + iris[0][0][2] + pupil[0][0][2], height - 1)
    min_x, max_x = 5, width - 1

    cv.rectangle(image_color, (min_x, min_y), (max_x, max_y), color=(255, 255, 200))

    show_results(image, image_color, text2="Segmentación de pupila, iris y artefactos")

    # Devolver las máscaras de las segmentaciones
    return fill_circle(image, pupil[0][0]), fill_circle(image, iris[0][0]), pupil_artifacts, iris_artifacts


# Aplica segment_eye a todas las imágenes del directorio test_images
def segment_test_images():
    # Get images from directory
    image_list = get_images('test_images')

    for image in image_list:
        segment_eye(image)


# Por defecto ejecuta la segmentación sobre todas las imágenes de prueba
segment_test_images()
