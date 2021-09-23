#!../../venv/bin/python

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST


if __name__ == '__main__':
    mndata = MNIST('../python-mnist/data')
    images, labels = mndata.load_training()

    X_all = np.array(images) / 255
    #  Y_all = [5 0 4 1 9...]
    Y_all = np.array(labels)
    # Y_all =
    # [[5 5 5 5 5 5 5 5 5 5 5]
    #  [0 0 0 0 0 0 0 0 0 0 0]
    #  [4 4 4 4 4 4 4 4 4 4 4]
    #  [1 1 1 1 1 1 1 1 1 1 1]
    #  [9 9 9 9 9 9 9 9 9 9 9]
    #  ...]
    Y_all = np.repeat(Y_all, 11).reshape(-1,11)
    #  Y_all =
    # [[0 0 0 0 0 1 0 0 0 0 0]
    #  [1 0 0 0 0 0 0 0 0 0 0]
    #  [0 0 0 0 1 0 0 0 0 0 0]
    #  [0 1 0 0 0 0 0 0 0 0 0]
    #  [0 0 0 0 0 0 0 0 0 1 0]
    #  ...]
    Y_all = (Y_all == np.arange(11)).astype(int)

    X_train, X_valid, X_test = np.split(X_all, (np.array([0.7, 0.9])*X_all.shape[0]).astype('int32'))
    Y_train, Y_valid, Y_test = np.split(Y_all, (np.array([0.7, 0.9])*Y_all.shape[0]).astype('int32'))

    W1 = np.random.randn(128, 784)
    b1 = np.random.randn(128)

    W2 = np.random.randn(11, 128)
    b2 = np.random.randn(11)

    for x, y_t in zip(X_train, Y_train):
        h1 = W1@x+b1
        z1 = np.maximum(0, h1)

        h2 = W2@z1 + b2
        y_p = np.exp(h2)/sum(np.exp(h2))

        loss = np.mean((y_t - y_p)**2)

        print(loss)
