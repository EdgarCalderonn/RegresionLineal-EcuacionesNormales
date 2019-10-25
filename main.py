# Trabajo realizado por Edgar Calderón - T00049682

import numpy as np

# Se cargan los datos del dataset
dataSet = np.loadtxt("fish_length.txt",delimiter='  ')


# En este desarrollo usaré 31 para entrenamiento, y 13 para probar - 70% / 30%

# Se crea la matriz x con los datos
x  = np.array(dataSet[0:31,[0,1]])

# Se crea nuestra matriz de soluciones
y  = np.array(dataSet[0:31,2])


# Primero se saca la inversa de el producto punto de x transpuesta y x
tetha = (np.linalg.inv(np.dot(x.transpose(),x)))


# Producto punto de la inversa con x transpuesta
tetha = np.dot(tetha,x.transpose())


# Producto punto de lo resultante con y
tetha = np.dot(tetha,y)




# Ahora usare el 40% de los datos
pruebax = np.array(dataSet[31:44,[0,1]])
pruebay =  np.array(dataSet[31:44,[2]])

cantidad = len(pruebax)

#Inicializo el vector con el numero de filas de prueba
predicciones = [0]*(cantidad)


# Aqui multiplico cada uno de los tetha[i] por x[i] en cada caso, esto es aplicar la formula, en este caso uso ciclos anidados, pero también se puede hacer con un solo ciclo, pero hay que escribir mas codigo
for i in range (0,cantidad):
    for j in range (2):
        predicciones[i] = predicciones[i] + tetha[j] * pruebax[i][j]
  
    #Aquí fuera del ciclo de j se muestra al usuario el valor real del precio y la prediccion  
    print("**PRUEBA "+str(i+1)+"**\nDato real: "+str(pruebay[i])+" \nPrediccion: "+str(predicciones[i])+"\nDiferencia: "+str(pruebay[i]-predicciones[i])+"\n\n")


    # Procedemos a calcular el error cuadratico para evaluar nuestras predicciones

# Se inicializan los valores en 0
error_cuadratico = 0
sumatoria = 0

# Se lleva a cabo la sumatoria 
for i in range (cantidad):
    sumatoria = sumatoria + (predicciones[i] - pruebay[i]) ** 2
    
error_cuadratico = (1/cantidad) * sumatoria

print("Error cuadrático medio: "+str(error_cuadratico[0]))

# El error medio se calcula con la raiz cuadrada del error cuadrado...
error_medio = np.sqrt(error_cuadratico)

print("El error promedio es: " + str(error_medio[0]) + "\n")
print("Por lo que vemos que nuestro modelo es decentemente acertado.")
    