from ex_1 import *
from ex_2 import *
from ex_3 import *
from ex_4 import *


####################################################
# Exercise 1 - adjustIntensity & equalizeIntensity #
####################################################
def equalize_test(do_plot=True):
    inImg = np.float32(cv.imread('examples/1/moon.png', cv.IMREAD_GRAYSCALE)/255)
    outImage = equalizeIntensity(inImg)
    print(outImage)
    if (do_plot):
        plot_2_with_hist(inImg, outImage)

def adjust_test(do_plot=True):
    inImg = np.float32(cv.imread('examples/1/bn_1.jpg', cv.IMREAD_GRAYSCALE)/255)
    outImage = adjustIntensity(inImg, outRange=[0.2,0.9])
    plot_2_with_hist(inImg, outImage)



###########################################################
# Exercise 2 - filterImage, gaussKernel1D, gaussianFilter #
###########################################################

def convol_test():
    # Ejemplo de las diapositivas
    inImg = np.uint8(([
    [45, 60, 98, 127, 132, 133, 137, 133],
    [46, 65, 98, 123, 126, 128, 131, 133],
    [47, 65, 96, 115, 119, 123, 135, 137],
    [47, 63, 91, 107, 113, 122, 138, 134],
    [50, 59, 80,  97, 110, 123, 133, 134],
    [49, 53, 68,  83,  97, 113, 128, 133],
    [50, 50, 58,  70,  84, 102, 116, 126],
    [50, 50, 52,  58,  69,  86, 101, 120]]))

    filtro = np.float32([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])
    outImage = filterImage(inImg, filtro)
    print(np.uint8(outImage))


# TODO: gaussianFilter, medianFilter
# Ejemplo ruido
# img = np.uint8(cv.imread('examples/Captura.jpeg', cv.IMREAD_GRAYSCALE))
# outImage = gaussianFilter(img, 2)
# cv.imwrite("outImage_ex2.jpg", outImage)


# img = np.uint8(cv.imread('examples/ruido_gaussiano.jpg', cv.IMREAD_GRAYSCALE))
# outImage = medianFilter(img, 3)
# cv.imwrite("outImage_ex2_median_3.jpg", outImage)
# outImage = medianFilter(img, 7)
# cv.imwrite("outImage_ex2_median_7.jpg", outImage)


# Example 1:
def erode1():
    img = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    SE = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    out = erode(img, SE)
    show_results(img, out)

# Example 2:
def example_j(option):
    match(option):
        case 'erode':
            img = np.uint8(cv.imread('examples/3/erode/j.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            SE = np.ones((5,5),np.uint8)
            out = erode(img, SE)

            show_results(img, out)
            img = np.uint8(cv.imread('examples/3/erode/j_result.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            show_results(img, out, global_title='Comparación', text1='Resultado esperado', text2='Resultado obtenido')
        case 'dilate':
            img = np.uint8(cv.imread('examples/3/dilate/j.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            SE = np.ones((5,5),np.uint8)
            out = dilate(img, SE)

            show_results(img, out)
            img = np.uint8(cv.imread('examples/3/dilate/j_result.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            show_results(img, out, global_title='Comparación', text1='Resultado esperado', text2='Resultado obtenido')
        case 'opening':
            img = np.uint8(cv.imread('examples/3/opening/j.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            SE = np.ones((5,5),np.uint8)
            out = opening(img, SE)

            show_results(img, out)
            img = np.uint8(cv.imread('examples/3/opening/j_result.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            show_results(img, out, global_title='Comparación', text1='Resultado esperado', text2='Resultado obtenido')
        case 'closing':
            img = np.uint8(cv.imread('examples/3/closing/j.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            SE = np.ones((5,5),np.uint8)
            out = closing(img, SE)

            show_results(img, out)
            img = np.uint8(cv.imread('examples/3/closing/j_result.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            show_results(img, out, global_title='Comparación', text1='Resultado esperado', text2='Resultado obtenido')


# Example 3: circles and lines
def circles_and_lines():
    se = [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]]
    
    img = np.uint(cv.imread('examples/3/opening/circles_lines.png', cv.IMREAD_GRAYSCALE))
    out = opening(img, se)

    show_results(img, out)
    img = np.uint8(cv.imread('examples/3/opening/circles_lines_result.png', cv.IMREAD_GRAYSCALE))
    img = np.where(img>100,1,0)
    show_results(img, out, global_title='Comparación', text1='Resultado esperado', text2='Resultado obtenido')

# Example 4: lines
def lines(option):
    img = np.uint8(cv.imread('examples/3/opening/lines.png', cv.IMREAD_GRAYSCALE))
    img = np.where(img>100,1,0)

    match(option):
        case 'vertical':
            SE = cv.getStructuringElement(cv.MORPH_RECT, (3, 9))
            out = opening(img, SE)

            show_results(img, out)
            img = np.uint8(cv.imread('examples/3/opening/lines_result_vert.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            show_results(img, out, global_title='Comparación', text1='Resultado esperado', text2='Resultado obtenido')
        case 'horizontal':
            SE = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
            out = opening(img, SE)

            show_results(img, out)
            img = np.uint8(cv.imread('examples/3/opening/lines_result_horiz.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            show_results(img, out, global_title='Comparación', text1='Resultado esperado', text2='Resultado obtenido')

# Example 5: cells
def cells(option):
    img = np.uint8(cv.imread('examples/3/opening/cells.png', cv.IMREAD_GRAYSCALE))
    img = np.where(img>100,1,0)

    match(option):
        case 11:
            se = [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]]
            out = opening(img, se)
            show_results(img, out)
            img = np.uint8(cv.imread('examples/3/opening/cells_result_11.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            show_results(img, out, global_title='Comparación', text1='Resultado esperado', text2='Resultado obtenido')
        case 7:
            se = [
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0] ]
            out = opening(img, se)
            show_results(img, out)
            img = np.uint8(cv.imread('examples/3/opening/cells_result_7.png', cv.IMREAD_GRAYSCALE))
            img = np.where(img>100,1,0)
            show_results(img, out, global_title='Comparación', text1='Resultado esperado', text2='Resultado obtenido')
            
def hit_or_miss_test():
    img = np.uint8(cv.imread('examples/3/hit-or-miss/image.png', cv.IMREAD_GRAYSCALE))
    img = np.where(img>100,1,0)
    img = invert(img, np.shape(img))

    hit = [
        [0,0,0],
        [0,1,0],
        [0,0,0]]
    miss = [
        [1,1,1],
        [1,0,1],
        [1,1,1]]
    out = hit_or_miss(img, hit, miss, (1,1))

    show_results(img, out)



##################################
# Exercise 4: gradientImage

def gradientImage_test(option):
    img = np.uint8(cv.imread('examples/4/gradientImage/in.jpg', cv.IMREAD_GRAYSCALE))
    [x,y] = gradientImage(img, option)

    # x,y = np.abs(x), np.abs(y)
    x,y = np.clip(x, 0, 255), np.clip(y, 0, 255)
    res = np.array(x + y)
    res = np.clip(res, 0, 255)
    show_results(x, y)



def log_test():
    img = np.uint8(cv.imread('examples/4/log/circles.png', cv.IMREAD_GRAYSCALE))
    sigma = 0.5
    res = LoG(img, sigma)

    show_results(img, res)

log_test()