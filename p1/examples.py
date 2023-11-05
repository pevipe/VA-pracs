from ex_3 import *


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
            
