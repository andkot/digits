#!../../venv/bin/python
import numpy as np

def f(x,y):
    return y*x

a = np.array([1,2,3,4])

b = [0,2,3,4]

c = f(a , b)

print(c)