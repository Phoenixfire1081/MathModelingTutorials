import numpy as np
import functools
from boundary import neumann  
from models import wvex 

t_max    = 4*np.pi 
x_max    = 4*np.pi 
num_grid = 61 
num_time = 8*61 

plot_conservative = False

def time(n = num_time):
    return np.linspace(0, t_max, n)

def grid(n = num_grid):
    return np.linspace(0, x_max, n)

def initial(x):
    # orography
    # heightu
    h = np.where((np.pi<x)&(x<3*np.pi), 0.25*(np.sin(x + np.pi*0.5) + 1.0),0) 
    # momenta
    m = 0*np.ones_like(x)
    m = 0.5*(m[1:] + m[:-1])
    return np.array((h,m))



boundary = neumann 
# bind gravity and coriolis force to shallow water solver constructor
# the resulting constructor will have signature solver(grid,boundary)
solver  = functools.partial(wvex.solver, boundary = boundary, func_c = lambda t: 1)
plotter = functools.partial(wvex.plotter,
                            content = range(2), 
                            gui={"nrows":1,"ncols":2,"num": 4}, bounds = [[-1.5,1.5],[-1.5,1.5]])