import numpy as np
import functools
from boundary import neumann
from models import shw

gravity  = 9.81
coriolis = lambda x: np.where(np.abs(22 - x) <= 10, 0.0 * np.pi * (1 - np.square(0.1 * (x - 22)) ), 0 ) 
t_max    = 5
x_max    = 40
num_grid = 51 
num_time = 101

plot_conservative = False 

def time(n = num_time):
    return np.linspace(0, t_max, n)

def grid(n = num_grid):
    return np.linspace(0, x_max, n)

def initial(x):
    #orography
    z  = np.where(np.abs(x - 20) <= 4, 2.48 * (1 - np.square(0.25 * (x - 20))),0)
    #height
    h  =  4 * np.ones_like(x)
    #momenta
    hu = 10 * np.ones_like(x)
    hv = np.zeros_like(x)
    initial = np.stack((h,hu,hv,z))
    # average using trapezoidal rule
    return 0.5*(initial[:,1:]+initial[:,0:-1])

boundary = neumann
# bind gravity and coriolis force to shallow water solver constructor
# the resulting constructor will have signature solver(grid,boundary)
solver  = functools.partial(shw.solver, 
                            boundary = boundary,
                            equation = shw.equation(gravity, coriolis))
plotter = functools.partial(shw.plotter,
                            content = [[0,3],1,2], 
                            boundary = boundary(2), 
                            gui={"nrows":2,"ncols":2,"num": 1})