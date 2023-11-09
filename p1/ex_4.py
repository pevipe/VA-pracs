from base import *
from ex_2 import filterImage


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

    img_x = filterImage(inImage, gx)
    img_y = filterImage(inImage, gy)
    
    return img_x, img_y


