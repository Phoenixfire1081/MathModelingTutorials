import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import numpy as np

def plot(grid, left, right):
    plt.figure(3)
    segs=np.zeros(shape=(grid.size-1,2,2));
    segs[:,0,0] = grid[:-1]
    segs[:,0,1] = left
    segs[:,1,0] = grid[1:]
    segs[:,1,1] = right 
    lc = mc.LineCollection(segs)
    plt.gca().add_collection(lc)
    plt.show()
