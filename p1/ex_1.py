from base import *



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



