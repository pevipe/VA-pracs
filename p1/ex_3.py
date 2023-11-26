from base import *


def calcWhiteCoords(SE, shape, center): # Calcula las coordenas en función del centro del SE con los valores a 1
    n,m = shape
    c_x, c_y = center
    coords = []
    for i in range(n):
        for j in range(m):
            if (SE[i][j]==1):
                coords.append((i-c_x, j-c_y))
    return coords


def findWhite(inImage):
    n,m = np.shape(inImage)
    coords = []
    for i in range(n):
        for j in range(m):
            if (inImage[i][j]==1): 
                coords.append((i,j))
    return coords


def erode(inImage, SE, center=[]):
    start_t = time()
    n,m = np.shape(inImage)
    outImage = np.zeros(np.shape(inImage), dtype=np.uint)

    if len(np.shape(SE)) == 1:
        SE_width = len(SE)
        SE_height = 1
        SE = np.reshape(SE, (1, len(SE)))   #Convertir en matriz para el bucle inferior
    else:
        SE_height, SE_width = np.shape(SE)
        

    if center==[]:
        center = SE_height//2+1, SE_width//2+1

    coords = calcWhiteCoords(SE, (SE_height, SE_width), center)

    # Establecer los límites inferior y derecho del bucle
    if (SE_height==1): to_i = n
    else: to_i = n-(SE_height-center[0]-1)

    if (SE_width==1): to_j = m
    else: to_j = m-(SE_width-center[1]-1)

    for i in range(center[0], to_i):
        for j in range(center[1], to_j):
            cumple = True
            for c2 in coords:
                check = i+c2[0], j+c2[1]
                if (inImage[check[0]][check[1]]==0):
                    cumple = False
                    break
            if (cumple):
                outImage[i][j] = 1

    print('Erode completed in', time()-start_t, 'seconds')
    return outImage


def dilate(inImage, SE, center=[]):
    start_t = time()
    n,m = np.shape(inImage)
    outImage = np.zeros(np.shape(inImage), dtype=np.uint)

    if len(np.shape(SE)) == 1:
        SE_width = len(SE)
        SE_height = 1
        SE = np.reshape(SE, (1, len(SE)))   #Convertir en matriz para el bucle inferior
    else:
        SE_height, SE_width = np.shape(SE)

    if center==[]:
        center = SE_height//2+1, SE_width//2+1

    coords = calcWhiteCoords(SE, (SE_height, SE_width), center)

    for i in range(n):
        for j in range(m):
            if (inImage[i][j]==1):
                for c2 in coords:
                    c = i+c2[0], j+c2[1]
                    if (c[0]>=0 and c[0]<n and c[1]>=0 and c[1]<m):
                        outImage[c] = 1

    print('Dilate completed in', time()-start_t, 'seconds')
    return outImage


def opening(inImage, SE, center=[]):
    tmp = erode(inImage, SE, center)
    return dilate(tmp, SE, center)


def closing(inImage, SE, center=[]):
    tmp = dilate(inImage, SE, center)
    return erode(tmp, SE, center)


def invert(object, shape):
    n,m = shape
    out = np.array(object, dtype=np.uint) + 1
    out = out % 2
    return out


def coherent_SE(objSE, bgSE):
    res = np.array(objSE, dtype=np.uint8) * np.array(bgSE, dtype=np.uint8)
    coherent = not res.any()
    return coherent


def hit_or_miss(inImage, objSE, bgSE, center=[]):
    start_t = time()
    assert(np.shape(objSE) == np.shape(bgSE)), "Las dimensiones de los EE no coinciden"
    
    if len(np.shape(objSE)) == 1:
        objSE = np.reshape(objSE, (1, len(objSE)))   # Convertir en matriz para el bucle inferior
        bgSE = np.reshape(bgSE, (1, len(bgSE)))   # Convertir en matriz para el bucle inferior

    # Comprobar incoherencia
    assert(coherent_SE(objSE, bgSE)), "Elementos estructurantes incoherentes"
    
    n, m = np.shape(inImage)

    invImg = invert(inImage, (n, m))

    im1 = erode(inImage, objSE, center)
    im2 = erode(invImg, bgSE, center)
    outImage = im1 * im2
    
    print('Hit_or_miss completed in', time()-start_t, 'seconds')
    return outImage

