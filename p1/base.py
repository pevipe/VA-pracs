import numpy as np
import cv2 as cv
from time import time
from matplotlib import pyplot as plt

# Para testing
def plot_2_with_hist(inImage, outImage):
    plt.figure(figsize=(12,6))

    # Imagen original
    plt.subplot(2,2,1)
    plt.imshow(inImage, cmap='gray')
    plt.title('Imagen original')
    plt.axis('off')

    # Histograma original
    plt.subplot(2,2,2)
    plt.hist(inImage.flatten(), bins=256, range=[0,1], color='gray')
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

def plot_comparison(img1, img2, global_title='Test', text1='Original', text2='Modificada'):
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