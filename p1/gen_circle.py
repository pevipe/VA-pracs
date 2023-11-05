import numpy as np
import matplotlib.pyplot as plt

# Definir el tamaño de la matriz
tamano = 31

# Crear una matriz llena de ceros
matriz = np.zeros((tamano, tamano), dtype=int)

# Calcular el centro del círculo
centro_x = tamano // 2
centro_y = tamano // 2

# Definir el diámetro del círculo
diametro = 31  # Asegúrate de que el diámetro sea el valor correcto

# Calcular el radio a partir del diámetro
radio = diametro // 2

# Llenar la matriz con "1" en los puntos que conforman el círculo
for i in range(tamano):
    for j in range(tamano):
        if (i - centro_x) ** 2 + (j - centro_y) ** 2 <= radio ** 2:
            matriz[i, j] = 1

# Imprimir la matriz
print(matriz)
print('[')
for i in range(np.shape(matriz)[0]):
    print("[", end="")
    for j in range(np.shape(matriz)[1]-1):
        print(str(matriz[i][j]) + ', ', end="")
    print(str(matriz[i][np.shape(matriz)[1]-1]) + '],')
print(']')

# Visualizar la matriz como una imagen
plt.imshow(matriz, cmap='gray', interpolation='nearest')
plt.show()
