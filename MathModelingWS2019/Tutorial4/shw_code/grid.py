import numpy as np

def extend_periodic(x,size):
    right = x[-1] - x[0] + x[1:size+1]
    left =  x[0] - x[-1] + x[-size-1:-1]
    return np.concatenate((left,x,right))

def inner(x,size):
    return x[size:-size]


def average(x,y):
    return (y[1:] + y[:-1])/2 #trapeziodal rule


