import numpy as np
import functools 
from boundary import periodic
from models import shw


gravity  = 9.81
coriolis = lambda x: 2 * np.pi
t_max    = 200
x_max    = 1
num_time = 19.87
num_grid = 201
plot_conservative = False
def time(n = num_time):
    return np.linspace(0, t_max, n)

def grid(n = num_grid):
    return np.linspace(0, x_max, n)

def initial(x):
    #orography
    # z = np.where((x<12) &(8<x), 0.2 - 0.05*np.square(x-10),0)
    z = 0.5 * (1 - 0.5 * (np.cos(np.pi * (x - 0.5) / 0.5) + 1))
    #height
    # h = 0.33 * np.ones_like(x)
    h = np.maximum(0, 0.4 - z + 0.04 * np.sin((x - 0.5) / 0.25) - np.maximum(0, -0.4 + z))
    #velocity
    u = 0 * x
    v = 0 * x
    initial = np.stack((h,h*u,h*v,z))
    # average using trapezoidal rule
    return 0.5*(initial[:,1:]+initial[:,0:-1])

boundary = periodic

# bind gravity and coriolis force to shallow water solver constructor
# the resulting constructor will have signature solver(grid,boundary)
solver  = functools.partial(shw.solver, 
                            boundary = boundary,
                            equation = shw.equation(gravity, coriolis))
plotter = functools.partial(shw.plotter,
                            content = [[0,3],1,2], 
                            boundary = boundary(2), 
                            gui={"nrows":2,"ncols":2,"num": 4})
