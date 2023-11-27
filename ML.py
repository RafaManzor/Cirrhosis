import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nrd
import heapq

def getCentroids(X_data,y_data):
  g = np.unique(y_data)

  fil = len(g)

  col = X_data.shape[1]

  c = np.empty((fil,col))

  #centroides
  for k in range(fil):
    m = np.where(y_data == g[k])
    Xa = X_data[m]

    c[k] = sum(Xa)/len(Xa)

  return(c,g)


def minkowski(x,y,p):
  return (sum(abs(x - y)**p))**(1/p)


def getDistances(x0, X_data, y_data):
  n = len(X_data)
  n_y = len(y_data)
  y_label = np.arange(n_y)
  dist = np.zeros(n)

  for i in range(n):
    dist[i] = minkowski(x0, X_data[i],2)

  return [dist,y_label]


def splitData(X,y,size_test = 0.2):
  N = len(X)
  i_arr = np.arange(N)
  nrd.shuffle(i_arr)
  N_test = int(N*size_test)

  X_test = X[i_arr[:N_test]]
  y_test = y[i_arr[:N_test]]

  X_train = X[i_arr[N_test:]]
  y_train = y[i_arr[N_test:]]
  return X_test, y_test, X_train, y_train


def accuracy(y_test, y_pred):
  N_test = len(y_test)
  acc = 0

  for i in range(N_test):
    if y_test[i] == y_pred[i]:
      acc += 1

  return acc/N_test


def getNeighbors(X, y, p, k):
  n = len(X)                      # Número de filas de X
  y_et = np.zeros(n,dtype=int)    # Vector para guardar las etiquetas ordenadas de clasificación
  k_vecinos = np.zeros(n)         # Vector para guardar las distancias ordenadas

  # Paso 1: Se obtienen las distancias del punto p a los puntos en X.
  distancias = getDistances(p, X, y)    # Obtenemos las distancias

  # Paso 2: Se obtienen los k vecinos más cercanos.
  indices = np.argsort(distancias[:,0])   # Obtenemos los índices de las distancias ya ordenadas
  for i in range(n):
    m = indices[i]           # Obtenemos el i-ésimo índice más pequeño
    k_vecinos[i] = distancias[m,0]     # Guardamos las distancias en el orden correspondiente (menor a mayor)
    y_et[i] = distancias[m,-1]         # Asignamos la etiqueta correspondiente

  return k_vecinos[:k], y_et[:k], indices[:k]  # Regresamos las k distancias más cercanas, sus respectivas etiquetas e índices en el conjunto de datos


def plotNeighbours(X, y, p, indices):
  plotPoints(X, y, color=True)      # Graficamos todos los puntos del conjunto de datos
  plt.plot(p[0],p[1], "ro", label="$P$")         # Graficamos el punto a clasificar
  for i in range(k):
    m = indices[i]        # Obtenemos el índice del i-ésimo punto más cercano
    plt.plot(X[m,0],X[m,-1], "k*")  # Graficamos el i-ésimo punto mas cercano
  plt.legend(loc="best")


def getPrediction(etiquetas):
  esp, rept = np.unique(etiquetas, return_counts=True) # Obtenemos las especies que hay en el vector de etiquetas y las veces que se repiten
  especie = np.argmax(rept) # Obtenemos el máximo en las repeticiones
  clase = esp[especie]      # Obtenemos el valor de la especie que se encuentra en la posición del máximo de repeticiones

  return clase


def knn(X_test, X_train, y_train, k):
  n = len(X_test) # Obtenemos el número de filas del conjunto de datos X_test
  y_new = np.zeros(n,dtype=int) # Creamos un vector de ceros para ir guardando las etiquetas asignadas
  for i in range(n):
    k_vecinos, y_et, indices = getNeighbors(X_train, y_train, X_test[i], k)   # Obtenemos los puntos más cercanos al punto i-ésimo de X_test
    y_new[i] = getPrediction(y_et)    # Hacemos la clasificación para el punto i-ésimo de X_test
  return y_new    # Regresamos el vector con las etiquetas predichas
