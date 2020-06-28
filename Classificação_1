import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import e
from scipy import optimize

# Importando os dados do problema
df = pd.read_excel(r'D:\ml\ex2data1.xlsx')

# Plotando o gráfico dos dados importados
g = sns.lmplot(x="Nota no Exame 1", y="Nota no Exame 2", data=df,  hue='Aprovado', fit_reg=False, legend_out=True)
# Novo título
new_title = ' '
g._legend.set_title(new_title)
# Novos legenda
new_labels = ['Não Aprovado', 'Aprovado']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
plt.show()


data = np.array(df.iloc[:,:])
X = data[:,0:2]
y = data[:,2:3]
(m, n) = X.shape


# Adicionar uma coluna de números 1
X_1=np.concatenate((np.ones((m,1)),np.array(X)),axis=1)

# Criando um theta inicial
theta_inicial = np.zeros([n + 1,1])

# Criando a função sigmoid
def sigmoid(X):

    sigmoid = 1.0 / (1.0 + e ** (-1.0 * X))

    return sigmoid

# Criando a função custo
def CostFunction(theta, X, y):
    
    m = X.shape[0]
    epsilon = 1e-5    

    h=sigmoid(X.dot(theta))
    J = (1./m) * ((-y.T).dot(np.log(h+epsilon)) - ((1-y).T).dot(np.log(1-h+epsilon)))
    grad = (1./m)*(X.T.dot((h - y)))
    return J

def teste(theta):
    return CostFunction(theta, X_1, y)

# Usando fmin para descobrir os valores mínimos da função
theta = optimize.fmin(teste, theta_inicial, maxiter=400)

# Plotando o gráfico com a "fronteira de decisão"
g = sns.lmplot(x="Nota no Exame 1", y="Nota no Exame 2", data=df,  hue='Aprovado', fit_reg=False, legend_out=True)
new_title = ' '
g._legend.set_title(new_title)
new_labels = ['Não Aprovado', 'Aprovado']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
# Ajustando os eixos
g.set(ylim=(30, 100))
g.set(xlim=(30, 100))
plt.plot([0,122.14],[125.17,0])
plt.show()

# Probabilidade de um aluno com 45 no Exame 1 e 85 no Exame 2 ser aprovado
notas=np.array([[1, 45, 85]])
prob = sigmoid(notas.dot(theta.T))
print('Para um estudante com 45 no Exame 1 e 85 no Exame 2, nós prevemos uma probabilidade de', prob, 'de ser aprovado.')

# Precisão do training set
def predict(theta, X):
    
    m, n = X.shape
    p = np.zeros(shape=(m, 1))

    h = sigmoid(X.dot(theta.T))

    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0

    return p

p = predict(theta, X_1)
print('Porcentagem de acertos no training set: ', np.mean(p==y)*100,'%')
