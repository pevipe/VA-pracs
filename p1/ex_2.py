from base import *

def filterImage(inImage, kernel):
    im_height, im_width = np.shape(inImage)

    # Tratar con kernels unidimensionales
    if len(np.shape(kernel)) == 1:
        kern_width = len(kernel)
        kern_height = 1
        kernel = kernel.reshape(1, -1)  # Convertir en matriz para operar
    else:
        kern_height, kern_width = np.shape(kernel)

    # Creamos una imagen de salida inicialmente llena de ceros
    out_x, out_y = im_width - (kern_width - 1), im_height - (kern_height - 1)
    outImage = np.zeros((out_y, out_x), dtype=np.float32)

    # Realizamos la convolución
    for y in range(out_y):
        for x in range(out_x):
            window = inImage[y:y+kern_height, x:x+kern_width]
            outImage[y,x] = np.sum(window*kernel)

    return outImage


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

