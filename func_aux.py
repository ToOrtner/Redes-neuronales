import numpy as np
import pandas as pd
import csv

M = 1  # Salida :  Maligno (T) , Benigno (F)
N = 10  # Entradas
P = 410  # instancias
p_train = 350
p_check = P - p_train

# los usamos para entrenar
entradas_train = np.random.normal(0, 0.5, (p_train, N))
tags_train = np.zeros((p_train, 1))

# se usan despues para verificar
entradas_check = np.random.normal(0, 0.5, (p_check, N))
tags_check = np.zeros((p_check, 1))

dataPath = 'tp1_ej1_training.csv'
dataViewer = pd.read_csv(dataPath, names=['Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

with open(dataPath) as csvfile:
    dataCSV = csv.reader(csvfile, delimiter=',')
    i = 0
    k = 0
    for row in dataCSV:

        if i < p_train:

            if row[0] == 'M':
                tags_train[i][0] = 1
            else:
                tags_train[i][0] = -1

            j = 0
            for col in range(1, len(row)):
                entradas_train[i][j] = row[col]
                j += 1
            i += 1
        else:
            if row[0] == 'M':
                tags_check[k][0] = 1
            else:
                tags_check[k][0] = -1

            j = 0
            for col in range(1, len(row)):
                entradas_check[k][j] = row[col]
                j += 1
            k += 1

def sumamatriz(x, y):
    res = x
    for i in range(1,len(y)):
        for j in range(len(y[i])):
            res[i][j] += y[i][j]
    return res

def bias_add(V):
    bias = -np.ones( (len(V),1) )
    return np.concatenate( (V,bias), axis=1)

def bias_sub( V):
    return V[:,:-1]

def activacionEscalon (entrada, w):
    return np.sign(np.dot(entrada, w))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def activacionSigmoidea(x,w):
    return sigmoid(np.dot(x,w))

def tangente(x,w):
    return np.tanh(np.dot(x,w))

def deriv_tangente(x):
    return 1 - np.square(x)

def deriv_sigm(x):
    return np.exp(-x)/(1 + np.exp(-x)) ** 2
