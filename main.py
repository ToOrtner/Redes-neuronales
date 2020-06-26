import numpy as np
import pandas as pd
import csv

# Primer problema: Cancer de Mamas/home/tomy/Desktop/Redes-tp1

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

def activacionSigmoidea(x, w):
    return sigmoid(np.dot(x, w))

def tangente(x,w):
    return np.tanh(np.dot(x,w))

def deriv_tangente(x):
    return 1 - np.square(x)

def deriv_sigm(x):
    return np.exp(-x)/(1 + np.exp(-x)) ** 2


# Clase capa de la red
# Es una base, podemos agregarle otros parametros por ahi
class redNeuronal():

    def __init__(self, dim_capas, f_activ, lr, batch_size):

        self.dim_capas = dim_capas
        self.entradas = dim_capas[0]
        self.f_activ = f_activ
        # dim_capas = [N,3,3,3,1]
        self.W = []  # primera pos que sea nula
        self.n = len(dim_capas)
        self.lr = lr  # learning rate
        self.B = batch_size
        self.deriv = deriv_tangente if f_activ == tangente else deriv_sigm
        self.error = []

        # creo las matrices de Weights
        self.W.append(0)  # primera capa no tiene wieghts asociados
        for k in range(1, self.n):
            ww = np.random.normal(0, 1, (self.dim_capas[k - 1] + 1, self.dim_capas[k]))
            self.W.append(ww)

    def fit(self, X, Z):

        H = np.random.permutation(X.shape[0])  # permuta con el numero de instancias
        for k in range(len(H)):
            h = H[k]
            Xh = X[h:h + self.B]
            Zh = Z[h:h + self.B]

            # feed forward
            Y = self.activation(Xh, self.B)

            # correccion de pesos con backpropagation (en cada batch)
            dW = self.backprop(Y, Zh)
            self.W = sumamatriz(self.W, dW)

    def activation(self, Xh, B):
        Y = []
        Y0 = np.zeros((B, self.dim_capas[0] + 1))
        Y0[:] = bias_add(Xh)
        Y.append(Y0)
        for i in range(1, self.n - 1):
            # creo todas las capas del medio con sus dims correspondientes
            Yi = np.zeros((B, self.dim_capas[i] + 1))
            Yi[:] = bias_add(self.f_activ(Y[i - 1], self.W[i]))  # funcion de act
            Y.append(Yi)
        Yn = np.zeros((B, self.dim_capas[-1]))  # capa de salida
        Yn[:] = self.f_activ(Y[-1], self.W[-1])  # ultima capa con ultimos Weights
        Y.append(Yn)
        return Y

    def backprop(self, Y, Z):

        dW = []
        dW.append(0)  # el primero no tiene weights
        # creo los dW con las mismas dimensiones que sus respectivos W para no tener conflicto de cuentas
        for i in range(1, self.n):
            dWi = np.zeros_like(self.W[i])

            dW.append(dWi)

        D = [0] * self.n  # lista vacia para los D's

        En = Z[0][0] - Y[-1][0][0]  # error de ultima capa
        dYn = self.deriv(Y[-1])[0][0]  # diferencial de funcion TANGENTE

        Dn = En * dYn
        D[-1] = Dn
        for k in range(self.n - 1, 0, -1):
            dW[k] = self.lr * np.dot(Y[k - 1].T, D[k])
            Ei = np.dot(D[k], self.W[k].T)
            dYi = self.deriv(Y[k - 1])
            Di = bias_sub(Ei * dYi)
            D[k] = Di

        self.error.append(np.mean(np.sum(np.square(En), axis=0)))

        return dW

    def predict(self, Input):

        return self.activation(Input, Input.shape[0])[-1]

    def accuracy(self, resultados, esperados):
        n = resultados.shape[0]
        aciertos = 0
        for i in range(n):
            if np.sign(resultados[i]) == np.sign(esperados[i]):
                aciertos += 1

        return aciertos / n
capas = [N, 7, 8, 1]
red = redNeuronal(capas,activacionSigmoidea, 0.05, 1)
red.fit(entradas_train,tags_train)
#print(entradas_check[1])
res = red.predict(entradas_check)
res

###fdssssssssssssssssssssssssss







    #capas
    Y0 = np.zeros( (1,S[0]+1) )
    Y1 = np.zeros( (1,S[1]+1) )
    Y2 = np.zeros( (1,S[2]+1) )
    Y3 = np.zeros( (1,S[3]) )
    
    W1 = np.random.normal( 0, 0.5, (S[0]+1, S[1]))
    W2 = np.random.normal( 0, 0.5, (S[1]+1, S[2]))
    W3 = np.random.normal( 0, 0.5, (S[2]+1, S[3]))
    
    
    #batch
    B = 1
    H = np.random.permutation(P)
    
    #learning rate y error por epoca
    lr = 0.3
    error = 0.0

    for i in range(len(H)):
        
        h = H[i]

        Xh = X[h:h+B]
        Zh = Z[h:h+B]
        
        #feed forward 
        #activacion tanh (sigmoidea)
        Y0[:] = bias_add(Xh)
        Y1[:] = bias_add( np.tanh(np.dot(Y0,W1)))
        Y2[:] = bias_add( np.tanh(np.dot(Y1,W2)))
        Y3[:] = np.tanh(np.dot(Y2,W3))

        dW1 = np.zeros_like(W1)
        dW2 = np.zeros_like(W2)
        dW3 = np.zeros_like(W3)
        
        #backpropagation
        
        E3 = Zh-Y3
        dY3 = 1-np.square(Y3)
        D3 = E3*dY3
        
        dW3 += lr * np.dot(Y2.T,D3)
        
        E2 = np.dot(D3, W3.T)
        dY2 = 1-np.square(Y2)
        D2 = bias_sub(E2*dY2)

        dW2 += lr * np.dot( Y1.T, D2)

        E1 = np.dot( D2, W2.T)
        dY1 = 1-np.square(Y1)
        D1 = bias_sub( E1*dY1)

        dW1 += lr * np.dot( Y0.T, D1)

        W1 += dW1
        W2 += dW2
        error += np.mean( np.sum( np.square(Zh-Y2),axis=1))