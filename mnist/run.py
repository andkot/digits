#!../../venv/bin/python

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST


# number of neurons on the first layer
nn1 = 128


def forward_propagation(X, Y, k='mse'):
    W1 = np.random.randn(nn1, 784)
    b1 = np.random.randn(nn1)

    W2 = np.random.randn(11, nn1)
    b2 = np.random.randn(11)

    l = []
    for x, y in zip(X, Y):
        h1 = W1 @ x + b1
        z1 = np.maximum(0, h1)

        h2 = W2 @ z1 + b2
        y_p = np.exp(h2) / sum(np.exp(h2))

        if k == 'mse':
            l.append(mse(y - y_p))
        elif k == 'cross_entropy':
            l.append(cross_entropy(y - y_p))

    return l


def mse(y, y_p):
    return np.mean((y - y_p) ** 2)


def cross_entropy(y, y_p):
    return np.mean(-y*np.log(y_p))


if __name__ == '__main__':
    mndata = MNIST('../python-mnist/data')
    images, labels = mndata.load_training()

    X_all = np.array(images) / 255
    Y_all = np.zeros((len(labels), 11))
    Y_all[range(len(Y_all)), np.array(labels)] = 1

    X_train, X_valid, X_test = np.split(X_all, (np.array([0.7, 0.9])*X_all.shape[0]).astype('int32'))
    Y_train, Y_valid, Y_test = np.split(Y_all, (np.array([0.7, 0.9])*Y_all.shape[0]).astype('int32'))

    loss = forward_propagation(X_train, Y_train)
    [print(l) for l in loss]
