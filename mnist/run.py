#!../../venv/bin/python

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST


if __name__ == '__main__':
    mndata = MNIST('../python-mnist/data')
    images, labels = mndata.load_training()

    print(np.array(images[0]).reshape(28,28))
    plt.imshow(np.array(images[0]).reshape(28, 28), 'gray')
    plt.show()