#!../../venv/bin/python
import numpy as np
from run import forward_propagation
nn1 = 128
# size output vector
nno = 11
# size input vector
nni = 784

x = np.random.rand(784)
y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

W1 = np.random.randn(nn1, nni)
b1 = np.random.randn(nn1)
W2 = np.random.randn(nno, nn1)
b2 = np.random.randn(nno)

h1, z1, h2, y_p = forward_propagation(x, W1, b1, W2, b2)

