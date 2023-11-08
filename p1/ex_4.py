import numpy as np
import cv2 as cv
from time import time

from matplotlib import pyplot as plt

from ex_2 import filterImage

def filterImage3(inImage, kernel):
    row, columns = inImage.shape
    p, q = kernel.shape

    padR = p//2
    padC = q//2
    padded = np.zeros( shape = (row+padR*2, columns+padC*2), dtype=inImage.dtype)
    padded[padR: padR + row, padC: padC + columns] = inImage

    outImg = np.zeros_like(inImage)

    for i in range(row):
        for j in range(columns):
            region = padded[i:i+p, j:j+q]
            result = np.sum(region*kernel)
            outImg[i,j] = result

    return outImg

def gradientImage (inImage, operator):

    match operator:
        case 'Roberts':
            gx = np.array([[-1,0], [0,1]])
            gy = np.array([[0,-1], [1,0]])
        case 'CentralDiff':
            gx = np.array([1,0,1])
            gy = np.array([[1],[0],[1]])
        case 'Prewitt':
            gx = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
            gy = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])
        case 'Sobel':
            gx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
            gy = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

    img_x = filterImage(inImage, gx)
    img_y = filterImage(inImage, gy)
    
    return img_x, img_y

img = np.uint8(cv.imread('examples/4/gradientImage/in.jpg', cv.IMREAD_GRAYSCALE))
# img = img

[x,y] = gradientImage(img, 'Roberts')

x,y = np.abs(x), np.abs(y)
# clip de x e y entre 0-255

res = np.array(x + y)
res = np.minimum(res,255)
res = np.abs(res)

def show_results(img1, img2, global_title='Test', text1='Original', text2='Modificada'):
    plt.figure(figsize=(12,6))
    plt.title(global_title)

    # Imagen original
    plt.subplot(1,2,1)
    plt.imshow(img1, cmap='gray')
    plt.title(text1)
    plt.axis('off')

    # Histograma original
    plt.subplot(1,2,2)
    plt.imshow(img2, cmap='gray')
    plt.title(text2)
    plt.axis('off')


    plt.tight_layout()
    plt.show()


show_results(img,res)