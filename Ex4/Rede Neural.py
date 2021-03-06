import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import fmin_cg
'''Credits for the user Benlau93 from GitHub for the most part of the code. I made some changes to make the optimization better.'''


# Importando dados
mat = loadmat('D:\ex4\ex4data1.mat')

X = mat["X"]
y = mat["y"]

# Visualizando os dados
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape(20,20,order="F"), cmap="binary") 
        axis[i,j].axis("off")
plt.show()

def sigmoid(z):
    
    return 1/ (1 + np.exp(-z))

# Computando o custo da rede neural
def nnCostFunction(nn_params,input_layer_size, hidden_layer_size, num_labels,X, y,Lambda):
    
    # Reshape nn_params de volta para os parâmetros Theta1 e Theta2
    Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)
    
    m = X.shape[0]
    J=0
    X = np.hstack((np.ones((m,1)),X))
    y10 = np.zeros((m,num_labels))
    
    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m,1)), a1)) # hidden layer
    a2 = sigmoid(a1 @ Theta2.T) # output layer
    
    for i in range(1,num_labels+1):
        y10[:,i-1][:,np.newaxis] = np.where(y==i,1,0)
    for j in range(num_labels):
        J = J + sum(-y10[:,j] * np.log(a2[:,j]) - (1-y10[:,j])*np.log(1-a2[:,j]))
    
    cost = 1/m* J
    reg_J = cost + Lambda/(2*m) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
    
    
    return reg_J

# Computando o gradiente da rede neural
def gradient(nn_params,input_layer_size, hidden_layer_size, num_labels,X, y,Lambda):
    # Reshape nn_params de volta para os parâmetros Theta1 e Theta2
    Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)
    # Implementando o algoritmo backpropagation para obter o gradiente  
    m = X.shape[0]
    
    X = np.hstack((np.ones((m,1)),X))
    grad1 = np.zeros((Theta1.shape))
    grad2 = np.zeros((Theta2.shape))
    y10 = np.zeros((m,num_labels))
    J=0
    
    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m,1)), a1)) # hidden layer
    a2 = sigmoid(a1 @ Theta2.T) # output layer

    for i in range(1,num_labels+1):
        y10[:,i-1][:,np.newaxis] = np.where(y==i,1,0)
    for j in range(num_labels):
        J = J + sum(-y10[:,j] * np.log(a2[:,j]) - (1-y10[:,j])*np.log(1-a2[:,j]))
    
    for i in range(m):
        xi= X[i,:] # 1 X 401
        a1i = a1[i,:] # 1 X 26
        a2i =a2[i,:] # 1 X 10
        d2 = a2i - y10[i,:]
        d1 = Theta2.T @ d2.T * sigmoidGradient(np.hstack((1,xi @ Theta1.T)))
        grad1= grad1 + d1[1:][:,np.newaxis] @ xi[:,np.newaxis].T
        grad2 = grad2 + d2.T[:,np.newaxis] @ a1i[:,np.newaxis].T
             
    grad1 = 1/m * grad1
    grad2 = 1/m*grad2
    
    grad1_reg = grad1 + (Lambda/m) * np.hstack((np.zeros((Theta1.shape[0],1)),Theta1[:,1:]))
    grad2_reg = grad2 + (Lambda/m) * np.hstack((np.zeros((Theta2.shape[0],1)),Theta2[:,1:]))
    grad = np.concatenate((grad1_reg.flatten(),grad2_reg.flatten()))

    return grad

# Gradiente da função sigmoid
def sigmoidGradient(z):
    
    sigmoid = 1/(1 + np.exp(-z))
    
    return sigmoid *(1-sigmoid)

# Inicializando os coeficientes de um layer com números aleatórios
def randInitializeWeights(L_in, L_out):
    
    epi = (6**1/2) / (L_in + L_out)**1/2
    
    W = np.random.rand(L_out,L_in +1) *(2*epi) -epi
    
    return W

# Inicializando algumas constantes do modelo
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10
Lambda = 1

# Inicializando os parâmetros gerados com número aleatórios
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.append(initial_Theta1.flatten(),initial_Theta2.flatten())

# Argumentos para a função de otimização
args = (input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

# Usando a função fmin_cg para encontrar os parâmetros que minimizam o erro e os seperando segundo seus layers
nnTheta=fmin_cg(nnCostFunction,x0=initial_nn_params, fprime=gradient, maxiter=100, args=args)
Theta1 = nnTheta[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
Theta2 = nnTheta[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

def predict(Theta1, Theta2, X):
    
    m= X.shape[0]
    X = np.hstack((np.ones((m,1)),X))
    
    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m,1)), a1)) # hidden layer
    a2 = sigmoid(a1 @ Theta2.T) # output layer
    
    return np.argmax(a2,axis=1)+1

pred3 = predict(Theta1, Theta2, X)
print("Precisão de classificação no training set:",sum(pred3[:,np.newaxis]==y)[0]/5000*100,"%")
