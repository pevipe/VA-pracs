import numpy as np
import cv2 as cv

img = cv.imread('bn_1.jpg')
assert img is not None, "file could not be read"
print(img.shape)

img2 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imwrite('bn_1_.jpg', img2)

px = img2[200, 100]
print(px)

# Lee la imagen bn_1.jpg, la convierte a escala de grises e imprime el valor asociado al píxel en la posición 200,100