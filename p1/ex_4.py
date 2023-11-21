from base import *
from ex_2 import filterImage, gaussianFilter


def gradientImage (inImage, operator):

    match operator:
        case 'Roberts':
            gx = np.array([[-1,0], [0,1]])
            gy = np.array([[0,-1], [1,0]])
        case 'CentralDiff':
            gx = np.array([1,0,1])
            gy = np.array([[1],[0],[1]])
        case 'Prewitt':
            gx = np.array([[-1/3,0,1/3], [-1/3,0,1/3], [-1/3,0,1/3]])
            gy = np.array([[-1/3,-1/3,-1/3], [0,0,0], [1/3,1/3,1/3]])
        case 'Sobel':
            gx = np.array([[-0.25,0,0.25], [-0.5,0,0.5], [-0.25,0,0.25]])
            gy = np.array([[-0.25,-0.5,-0.25], [0,0,0], [0.25,0.5,0.25]])
        case _:
            raise Exception('The operator does not match any of the possibles')

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

