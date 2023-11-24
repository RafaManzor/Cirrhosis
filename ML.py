import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy.random as nrd

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


def centroide(X,y,X_new):
  X_centro, y_centro = getCentroids(X,y)

  Nnew = len(X_new)
  y_pred = np.full(Nnew,-1)

  for i in range(Nnew):
      dist = getDistances(X_new[i],X_centro, y_centro)

      dist.sort(key = lambda pair: pair[0])
      y_pred[i] = np.argmin(dist[1])

  return y_pred


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


# Graficar puntos y centroides
# Entrada: datos - X_data, centroides - X_centroids
# Salida:  gráfica los puntos con los centroides
def plotCentroids( X_data, y_data, X_centroids, y_centroids):
    labels = np.unique(y_data)  # conjunto de etiquetas
    clases =  ['setosa','versicolor','virginica']
    colors  = ['tab:red', 'tab:green', 'tab:blue']
    markers = ['.', '.', '.']

    for k in labels:
        # seleccionar filas etiquetadas con la k-ésima categoría
        X_points = X_data[ y_data == k]
        plt.plot( X_points[:,0], X_points[:,1], ".", color=colors[k])

        # graficar centroids
        X_centro = X_centroids[ y_centroids == k][0]
        plt.plot( X_centro[0], X_centro[1], "o", color=colors[k], label=clases[k] )

        plt.legend(loc="best")
        
        
def assingCentroid( X_centro, y_centro, X_test ):
    N_test   = len(X_test)                                # número de puntos nuevos
    y_pred = np.full(N_test, -1)                          # arreglo de predicciones

    for i in range(N_test):
        # obtener distancias hacia los centroides
        dist = getDistances(X_test[i], X_centro, y_centro)

        # ordenar los pares (etiqueta, distancia) en orden creciente de acuerdo con la distancia
        dist.sort( key = lambda pair: pair[0])

        # asignar la etiqueta del centroide más cercano
        y_pred[i] = dist[0][1]

    return y_pred
  
  
  
  
# Graficar puntos y centroides
def plotPointsCentroids( X_points, X_centroids, y_centroids):
    # graficar todos los puntos con "."
    plt.plot( X_points[:,0], X_points[:,1], "." )

    # graficar centroides con "o"
    for x, y in zip(X_centroids, y_centroids):
        plt.plot( x[0], x[1], "o", label="Clase " + str(y))

    plt.legend(loc="best")
    
    
    
    
def randomCentroids(X,k):
  N = len(X)
  filas = np.arange(N)
  k_filas = nrd.choice(filas,k,replace = False)
  X_centroids = np.copy(X[k_filas])
  y_centroids = np.arange(k)
  return X_centroids, y_centroids
  
  
  
# Graficar puntos por clase o sin clase
def plotPoints( X_data, y_data, color=True ):
    if color == True:
        labels = np.unique(y_data)  # conjunto de etiquetas
        for k in labels:
            # seleccionar filas etiquetadas con la k-ésima categoría
            X_points = X_data[ y_data == k]
            plt.plot( X_points[:,0], X_points[:,1], ".", label=k )

            plt.legend(loc="best")
    else:
        plt.plot( X_data[:,0], X_data[:,1], ".")
        
        
        
def init_centroids(X,k):
  N,n = X.shape
  X_centro = np.zeros([k,n])
  y_centro = np.arange(k)


  #primer centroid
  X_centro[0] = X[nrd.randint(0,N+1)]
  #Calcular k-1 centroides restantes
  for i in range(1,k):
    dist = np.zeros(N)
    for p in range(N):
      min_d = 100000

      for j in range(i): #ciclo para centroides
        d = minkowski(X[p], X_centro[j],2)
        min_d = d

      dist[p] = min_d

    X_centro[i] = X[i]

  return X_centro, y_centro
    
    

def Kmeans(X,k,MAXITE):
  X_centro, y_centro = randomCentroids(X,k)

  for i in range(MAXITE):
    y_pred = assingCentroid(X_centro, y_centro, X)

    X_centroids, y_centroids = getCentroids(X, y_pred)

  return y_pred
  
  

def Kmeans_pp(X,k,MAXITE):
  X_centro, y_centro = init_centroids(X,k)

  for i in range(MAXITE):
    y_pred = assingCentroid(X_centro, y_centro, X)

    X_centroids, y_centroids = getCentroids(X, y_pred)

  return y_pred