#!../../venv/bin/python

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST


# number of neurons on the first layer
from sklearn.utils import shuffle

nn1 = 128
nno = 11
nni = 784
n_epoch = 10
h = 0.01
n_batches = 10


def forward_propagation(x, W1, b1, W2, b2):
    h1 = W1 @ x + b1
    z1 = np.maximum(0, h1)

    h2 = W2 @ z1 + b2
    y_p = np.exp(h2) / sum(np.exp(h2))

    return h1, z1, h2, y_p


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
    Y_all = np.zeros((len(labels), nno))
    Y_all[range(len(Y_all)), np.array(labels)] = 1

    X_train, X_valid, X_test = np.split(X_all, (np.array([0.1, 0.9])*X_all.shape[0]).astype('int32'))
    Y_train, Y_valid, Y_test = np.split(Y_all, (np.array([0.1, 0.9])*Y_all.shape[0]).astype('int32'))


    """"""""""""""""""""""""""""""""""""""""""
    W1 = np.random.randn(nn1, nni)
    b1 = np.random.randn(nn1)
    W2 = np.random.randn(nno, nn1)
    b2 = np.random.randn(nno)

    for i in range(n_epoch):
        X, Y = shuffle(X_train, Y_train)
        X_batches = np.split(X, n_batches)
        Y_batches = np.split(Y, n_batches)

        for X, Y in zip(X_batches, Y_batches):
            # массивы производных
            dW1 = np.zeros((nn1, nni))
            db1 = np.zeros(nn1)
            dW2 = np.zeros((nno, nn1))
            db2 = np.zeros(nno)

            # счиатю производные для каждой картинки в батче и добовляю их к массивам произвдодных
            for x, y in zip(X, Y):
                h1, z1, h2, y_p = forward_propagation(x, W1, b1, W2, b2)
                for i, j in zip(np.arange(nn1), np.arange(nni)):
                    dW1[i, j] =dW1[i, j] + int(h1[i] > 0) * x[j] * sum([(y_p[k] - y[k]) * W1[k, i] for k in range(nno)]) / nno
                    db1[i] = db1[i] + int(h1[i] > 0) * sum([(y_p[k] - y[k]) * W1[k, i] for k in range(nno)]) / nno
                for k, i in zip(np.arange(nno), np.arange(nn1)):
                    dW2[k, i] = dW2[k, i] + (y_p[k] - y[k]) * z1[i] / nno
                    db2[k] = db2[k] + (y_p[k] - y[k]) / nno

            # считаю среднюю производную по всем картинкам в батче
            dW1 = dW1 / X.shape[0]
            db1 = db1 / X.shape[0]
            dW2 = dW2 / X.shape[0]
            db2 = db2 / X.shape[0]

            # изменяю веса по градиентному спуску
            W1 = W1 - h*dW1
            b1 = b1 - h*db1
            W2 = W2 - h*dW2
            b2 = b2 - h*db2




