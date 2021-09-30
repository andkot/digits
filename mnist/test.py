#!../../venv/bin/python
import numpy as np

# number of neurons on the first layer
from run import forward_propagation

nn1 = 128
# size output vector
nno = 11
# size input vector
nni = 784
n_epoch = 10
h = 0.01
n_batches = 10

x = np.random.rand(784)
y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

W1 = np.random.randn(nn1, nni)
b1 = np.random.randn(nn1)
W2 = np.random.randn(nno, nn1)
b2 = np.random.randn(nno)

h1, z1, h2, y_p = forward_propagation(x, W1, b1, W2, b2)
""""""""""""""""""""""""""""""""""""""""""
dW1 = np.zeros((nn1, nni))
db1 = np.zeros(nn1)
dW2 = np.zeros((nno, nn1))
db2 = np.zeros(nno)

print('--------------------')

dW1 = np.zeros((nn1, nni))
db1 = np.zeros(nn1)
dW2 = np.zeros((nno, nn1))
db2 = np.zeros(nno)

db2 = db2 + (y_p - y)/nno
dW2 = dW2 + (y_p - y).reshape((-1, 1)) @ z1.reshape((1, -1))/nno
db1 = db1 + (h1>0)*((y_p-y)@W2)/nno
dW1 = dW1 + ((h1>0)*((y_p-y)@W2)).reshape(-1,1)@x.reshape(1,-1)/nno


print(dW1)
