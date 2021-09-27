#!../../venv/bin/python

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST


# number of neurons on the first layer
nn1 = 128


def forward_propagation(x, W1, b1, W2, b2):
    h1 = W1 @ x + b1
    z1 = np.maximum(0, h1)

    h2 = W2 @ z1 + b2
    y_p = np.exp(h2) / sum(np.exp(h2))

    d = {'h1': h1, 'z1': z1, 'h2': h2, 'y-p': y_p}

    return d


def backward_propagation(h1, z1, h2, y_p):
    pass


def loss(y, y_p, kind):
    return globals()[kind](y, y_p)


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

    W1 = np.random.randn(nn1, 784)
    b1 = np.random.randn(nn1)

    W2 = np.random.randn(11, nn1)
    b2 = np.random.randn(11)