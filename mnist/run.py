#!../../venv/bin/python

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
from sklearn.utils import shuffle

# number of neurons on the first layer
NN1 = 128
# size output vector
NN0 = 11
# size input vector
NNI = 784
N_EPOCH = 10
H = 0.01
N_BATCHES = 10


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
    Y_all = np.zeros((len(labels), NN0))
    Y_all[range(len(Y_all)), np.array(labels)] = 1

    X_train, X_valid, X_test = np.split(X_all, (np.array([0.1, 0.9])*X_all.shape[0]).astype('int32'))
    Y_train, Y_valid, Y_test = np.split(Y_all, (np.array([0.1, 0.9])*Y_all.shape[0]).astype('int32'))


    """"""""""""""""""""""""""""""""""""""""""
    W1 = np.random.randn(NN1, NNI)
    b1 = np.random.randn(NN1)
    W2 = np.random.randn(NN0, NN1)
    b2 = np.random.randn(NN0)

    list_loss_valid =[]
    list_loss_train =[]

    for i in range(N_EPOCH):
        X, Y = shuffle(X_train, Y_train)
        X_batches = np.split(X, N_BATCHES)
        Y_batches = np.split(Y, N_BATCHES)

        for X, Y in zip(X_batches, Y_batches):
            # массивы производных
            dW1 = np.zeros((NN1, NNI))
            db1 = np.zeros(NN1)
            dW2 = np.zeros((NN0, NN1))
            db2 = np.zeros(NN0)

            # счиатю производные для каждой картинки в батче и добовляю их к массивам произвдодных
            for x, y in zip(X, Y):
                h1, z1, h2, y_p = forward_propagation(x, W1, b1, W2, b2)
                db2 = db2 + (y_p - y) / NN0
                dW2 = dW2 + (y_p - y).reshape((-1, 1)) @ z1.reshape((1, -1)) / NN0
                db1 = db1 + (h1 > 0) * ((y_p - y) @ W2) / NN0
                dW1 = dW1 + ((h1 > 0) * ((y_p - y) @ W2)).reshape(-1, 1) @ x.reshape(1, -1) / NN0

            # считаю среднюю производную по всем картинкам в батче
            dW1 = dW1 / X.shape[0]
            db1 = db1 / X.shape[0]
            dW2 = dW2 / X.shape[0]
            db2 = db2 / X.shape[0]

            # изменяю веса по градиентному спуску
            W1 = W1 - H * dW1
            b1 = b1 - H * db1
            W2 = W2 - H * dW2
            b2 = b2 - H * db2

        # это для картинки
        YP_valid = []
        for x in X_valid:
            h1, z1, h2, y_p = forward_propagation(x, W1, b1, W2, b2)
            YP_valid.append(y_p)
        list_loss_valid.append(np.mean(loss(Y_valid, YP_valid, 'cross_entropy')))

        YP_train = []
        for x in X_train:
            h1, z1, h2, y_p = forward_propagation(x, W1, b1, W2, b2)
            YP_train.append(y_p)
        list_loss_train.append(np.mean(loss(Y_train, YP_train, 'cross_entropy')))

    plt.scatter(range(10), list_loss_valid, label='valid')
    plt.scatter(range(10), list_loss_train, label='train')
    plt.legend()
    plt.show()
