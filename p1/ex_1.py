import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt


def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):

    if (inRange==[]):
        inRange = [np.min(inImage), np.max(inImage)]

    # Evitamos valores imposibles
    inRange = [max([inRange[0], 0]), min([1, inRange[1]])]
    outRange = [max([outRange[0], 0]), min([1, outRange[1]])]

    # Calculamos la imagen de salida y la devolvemos
    factor = (outRange[1]-outRange[0]) / (inRange[1]-inRange[0])
    image = outRange[0] + factor* (inImage-inRange[0])

    return image

def equalizeIntensity (inImage, nBins=256):

    histograma = cv.calcHist([inImage], [0], None, [256], [0, 1]) # Sobre 256
    n,m = np.shape(inImage)

    hist_acumulado = np.cumsum(histograma)
    hist_acumulado = (hist_acumulado/(n*m))*(nBins-1)
    hist_acumulado = hist_acumulado.astype(np.uint8)
    
    imagen_ecualizada = np.float32(cv.LUT((inImage*255).astype(np.uint8), hist_acumulado/nBins))
    
    return imagen_ecualizada



# # # # # # # # # # # # 
# EJECUCIÓN DE EJEMPLO #
# # # # # # # # # # # #


# Leemos la imagen deseada (estandarizándola)
img_name = 'examples/moon.png'
img = np.float32(cv.imread(img_name, cv.IMREAD_GRAYSCALE)/255)

outImage = equalizeIntensity(img)



# Guardamos la imagen 
cv.imwrite("outImage.jpg", outImage*255)


# # # # # # # # # # # # # # # # # #
# VISUALIZACIÓN DE LOS RESULTADOS #
# # # # # ## # # # # ## # # # # # #

def plotGraphs():
    plt.figure(figsize=(12,6))

    # Imagen original
    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagen original')
    plt.axis('off')

    # Histograma original
    plt.subplot(2,2,2)
    plt.hist(img.flatten(), bins=256, range=[0,1], color='gray')
    plt.xlabel('Valor de píxel')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de la imagen original')

    # Imagen modificada
    plt.subplot(2,2,3)
    plt.imshow(outImage, cmap='gray', vmin=0, vmax=1) #
    plt.title('Imagen modificada')
    plt.axis('off')

    # Histograma de la imagen modificada
    plt.subplot(2,2,4)
    plt.hist(outImage.flatten(), bins=256, range=[0,1], color='gray')
    plt.xlabel('Valor de píxel')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de la imagen modificada')


    plt.tight_layout()
    plt.show()

# Opcional
plotGraphs()