#simple test case to check for bugs etc.
import numpy as np
import functools
from boundary import periodic 
from models import shw
gravity  = 9.81
coriolis = lambda x: 0 
t_max    = 10 
x_max    = 10 
num_grid = 21 
num_time = 51

plot_conservative = False

def time(n = num_time):
    return np.linspace(0, t_max, n)

def grid(n = num_grid):
    return (np.linspace(0, x_max, n))

def initial(x):
    x=0.5*(x[1:]+x[:-1])
    # orography
    z = 0*np.where((4<x)&(x<6) ,1.25,0.0)
    # heightu
    h  = np.maximum(1.0-z ,0.0)
    h = np.where((4<x)&(x<6) ,1.5,0.5)
    # momenta
    hu = np.zeros_like(x)
    hv = np.zeros_like(x)
    initial = np.stack((h,hu,hv,z))
    # average using trapezoidal rule
    return initial 


boundary = periodic 
# bind gravity and coriolis force to shallow water solver constructor
# the resulting constructor will have signature solver(grid,boundary)
solver  = functools.partial(shw.solver, 
                            boundary = boundary,
                            equation = shw.equation(gravity, coriolis))
plotter = functools.partial(shw.plotter,
                            content = [[0,3],1,2], 
                            boundary = boundary(2), 
                            conservative = plot_conservative,
                            labels = ["height","velocity x","velocity y"],
                            bounds = [[0,1.5]],
                            gui={"nrows":2,"ncols":2,"num": 1})