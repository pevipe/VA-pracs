from base import *
from ex_2 import filterImage, gaussianFilter


def gradientImage(inImage, operator):
    match operator:
        case 'Roberts':
            gx = np.array([[-1, 0], [0, 1]])
            gy = np.array([[0, -1], [1, 0]])
        case 'CentralDiff':
            gx = np.array([1, 0, 1])
            gy = np.array([[1], [0], [1]])
        case 'Prewitt':
            gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        case 'Sobel':
            gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, -1]])
            gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        case _:
            raise Exception('The operator does not match any of the options')

    img_x = filterImage(inImage, gx)
    img_y = filterImage(inImage, gy)

    return img_x, img_y


def cruces_por_cero(inImage):
    res = np.zeros(np.shape(inImage), dtype=np.uint8)

    for x in range(1, np.shape(inImage)[0] - 1):
        for y in range(1, np.shape(inImage)[1] - 1):
            if inImage[x, y] > 0:
                if (inImage[x - 1, y] < 0 or inImage[x + 1, y] < 0 or
                        inImage[x, y - 1] < 0 or inImage[x, y + 1] < 0):
                    res[x, y] = 1
    return res


def LoG(inImage, sigma):
    gaussFiltered = gaussianFilter(inImage, sigma)
    laplacian_kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    laplacian_over_gaussian = filterImage(gaussFiltered, laplacian_kernel)

    return cruces_por_cero(laplacian_over_gaussian)


def simplificar_direcciones(dirs):
    angles = dirs * (180.0 / np.pi)
    angles[angles < 0] += 180

    for i in range(np.shape(angles)[0]):
        for j in range(np.shape(angles)[1]):
            if 22.5 <= angles[i, j] < 67.5:
                angles[i, j] = 45
            elif 67.5 <= angles[i, j] < 112.5:
                angles[i, j] = 90
            elif 112.5 <= angles[i, j] < 157.5:
                angles[i, j] = 135
            else:
                angles[i, j] = 0

    return angles


def non_max_suppression(magnitude, direction):
    n, m = np.shape(magnitude)
    # Inicializar la imagen resultante después de la supresión no máxima
    result = np.zeros([n, m], dtype=np.uint8)

    for i in range(1, n - 1):
        for j in range(1, m - 1):
            prev = post = 255

            if direction[i, j] == 0:
                prev = magnitude[i, j - 1]
                post = magnitude[i, j + 1]
            elif direction[i, j] == 45:
                prev = magnitude[i + 1][j - 1]
                post = magnitude[i - 1][j + 1]
            elif direction[i, j] == 90:
                prev = magnitude[i - 1][j]
                post = magnitude[i + 1][j]
            elif direction[i, j] == 135:
                prev = magnitude[i - 1][j - 1]
                post = magnitude[i + 1][j + 1]

            if magnitude[i, j] > prev and magnitude[i, j] > post:
                result[i, j] = magnitude[i, j]

    return result


def histeresis(inImage, tlow, thigh, norms):
    n, m = np.shape(inImage)
    result = np.zeros([n, m], dtype=np.uint8)
    visited = set()

    for i in range(1, n - 1):
        for j in range(1, m - 1):
            if (i, j) in visited: continue

            if inImage[i, j] > thigh or result[i, j] == 255:
                result[i, j] = 255

                # Recorrer dirección del borde
                match norms[i, j]:
                    case 0:
                        check1 = i, j - 1
                        check2 = i, j + 1
                    case 45:
                        check1 = i + 1, j - 1
                        check2 = i - 1, j + 1
                    case 90:
                        check1 = i - 1, j
                        check2 = i + 1, j
                    case _:  # 135
                        check1 = i - 1, j - 1
                        check2 = i + 1, j + 1

                if inImage[check1] > tlow: result[check1] = 255
                if inImage[check2] > tlow: result[check2] = 255

                # Una vez se comprueban sus vecinos de acuerdo con las direcciones, ese píxel no tiene que
                # volver a comprobarse
                visited.add((i, j))
    return result


def edgeCanny(inImage, sigma, tlow, thigh):
    gauss_filtered = gaussianFilter(inImage, sigma)
    borders_x, borders_y = gradientImage(gauss_filtered, "Sobel")

    magnitude = np.sqrt(borders_x ** 2 + borders_y ** 2)
    direction = np.arctan2(borders_y, borders_x)

    dirs_simplified = simplificar_direcciones(direction)
    norms = dirs_simplified + 90 % 180

    non_max_supressed = non_max_suppression(magnitude, dirs_simplified)
    calculada_histeresis = histeresis(non_max_supressed, tlow, thigh, norms)

    return calculada_histeresis
