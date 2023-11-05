import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt


# def filterImage (inImage, kernel):
#     im_height, im_width     = np.shape(inImage)

#     # Tratar con kernels unidimensionales
#     if (len(np.shape(kernel))==1):
#         kern_width  = len(kernel)
#         kern_height = 1
#         kernel = kernel.reshape(1,-1) # Convertir en matriz para operar
#     else:
#         kern_height, kern_width = np.shape(kernel)

#     kern_center_x = kern_width//2  + 1
#     kern_center_y = kern_height//2 + 1

#     # Creamos una imagen de salida inicialmente llena de ceros
#     out_x, out_y = im_height-(kern_height-1), im_width-(kern_width-1)
#     outImage = np.zeros((out_y, out_x), dtype=np.uint8)

#     # Realizamos la convolución
#     for y in range(out_y):
#         for x in range(out_x):
#             px = 0
#             for ky in range(kern_height):
#                 for kx in range(kern_width):
#                     # Coordenadas en la imagen de entrada
#                     img_x = x - (kx - kern_center_x)
#                     img_y = y - (ky - kern_center_y)

#                     # Comprobamos que las coordenadas estén dentro de los límites de la imagen
#                     if img_x >= 0 and img_x < im_width and img_y >= 0 and img_y < im_height:
#                         px += inImage[img_y, img_x] * kernel[ky, kx]
#             outImage[y, x] = round(px)

#     return outImage


def filterImage(inImage, kernel):
    im_height, im_width = np.shape(inImage)

    # Tratar con kernels unidimensionales
    if len(np.shape(kernel)) == 1:
        kern_width = len(kernel)
        kern_height = 1
        kernel = kernel.reshape(1, -1)  # Convertir en matriz para operar
    else:
        kern_height, kern_width = np.shape(kernel)

    kern_center_x = kern_width // 2
    kern_center_y = kern_height // 2

    # Creamos una imagen de salida inicialmente llena de ceros
    out_x, out_y = im_width - (kern_width - 1), im_height - (kern_height - 1)
    outImage = np.zeros((out_y, out_x), dtype=np.uint8)

    # Realizamos la convolución
    for y in range(out_y):
        for x in range(out_x):
            px = 0
            for ky in range(kern_height):
                for kx in range(kern_width):
                    # Coordenadas en la imagen de entrada
                    img_x = x + kx - kern_center_x
                    img_y = y + ky - kern_center_y

                    # Comprobamos que las coordenadas estén dentro de los límites de la imagen
                    if img_x >= 0 and img_x < im_width and img_y >= 0 and img_y < im_height:
                        px += inImage[img_y, img_x] * kernel[ky, kx]
            outImage[y, x] = np.uint8(px)


    return outImage


### Ejemplo de las diapositivas

img = np.uint8(([[45, 60, 98, 127, 132, 133, 137, 133],
[46, 65, 98, 123, 126, 128, 131, 133],
[47, 65, 96, 115, 119, 123, 135, 137],
[47, 63, 91, 107, 113, 122, 138, 134],
[50, 59, 80, 97, 110, 123, 133, 134],
[49, 53, 68, 83, 97, 113, 128, 133],
[50, 50, 58, 70, 84, 102, 116, 126],
[50, 50, 52, 58, 69, 86, 101, 120]]))

filtro = np.float32([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])

# outImage = filterImage(img, filtro)
# print(np.uint8(outImage))

### 

# img = np.float32(cv.imread('bn_1.jpg', cv.IMREAD_GRAYSCALE)/255)
# definir filtro y llamar a la función
# cv.imwrite("outImage_ex2.jpg", outImage*255)

# def gaussKernel1D(sigma):
#     # Calcular el kernel
#     kern_N = int(2*np.ceil(3*sigma)+1)
#     # kern_center = kern_N // 2 +1
#     kernel = np.zeros(kern_N, np.float32)
#     for x in range(kern_N):
#         kernel[x] = np.exp(- (x / 2*sigma) ** 2) / (np.sqrt(2*np.pi)*sigma)
#     kernel = kernel / np.sum(kernel) # Normalizar el kernel

#     return kernel

def gaussKernel1D(sigma):
    # Calcular el kernel
    kern_N = int(2 * np.ceil(3 * sigma) + 1)
    kernel = np.zeros(kern_N, dtype=np.float32)
    kern_center = kern_N // 2
    for x in range(kern_N):
        kernel[x] = np.exp(- (x - kern_center) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    kernel = kernel / np.sum(kernel)  # Normalizar el kernel

    return kernel


def gaussianFilter(inImage, sigma):
    kernel = gaussKernel1D(sigma)
    tmp = filterImage(inImage, kernel)
    outImage = filterImage(tmp, kernel.reshape(-1,1))
    return outImage



# Ejemplo ruido
# img = np.uint8(cv.imread('examples/Captura.jpeg', cv.IMREAD_GRAYSCALE))
# outImage = gaussianFilter(img, 2)
# cv.imwrite("outImage_ex2.jpg", outImage)



def medianFilter(inImage, filterSize):
    im_height, im_width = np.shape(inImage)

    # Creamos una imagen de salida inicialmente llena de ceros
    out_x, out_y = im_width - (filterSize - 1), im_height - (filterSize - 1)
    outImage = np.zeros((out_y, out_x), dtype=np.uint8)

    # Aplicamos el filtro
    for y in range(out_y):
        for x in range(out_x):
            kernel = np.zeros(filterSize * filterSize)
            for ky in range(filterSize):
                for kx in range(filterSize):
                    img_x = x + kx
                    img_y = y + ky
                    kernel[ky * filterSize + kx] = inImage[img_y, img_x]
            px = np.uint8(np.median(kernel))
            outImage[y, x] = px

    return outImage



img = np.uint8(cv.imread('examples/ruido_gaussiano.jpg', cv.IMREAD_GRAYSCALE))
outImage = medianFilter(img, 3)
cv.imwrite("outImage_ex2_median_3.jpg", outImage)
outImage = medianFilter(img, 7)
cv.imwrite("outImage_ex2_median_7.jpg", outImage)