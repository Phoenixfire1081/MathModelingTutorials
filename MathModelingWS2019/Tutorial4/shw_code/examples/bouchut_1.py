import numpy as np
import functools 
from boundary import neumann
from models import shw


gravity  = 9.81
coriolis = lambda x: 2*np.pi
t_max    = 100
x_max    = 25 
num_time = 50
num_grid = 201
plot_conservative = False
def time(n = num_time):
    return np.linspace(0, t_max, n)

def grid(n = num_grid):
    return np.linspace(0, x_max, n)

def initial(x):
    #orography
    z = np.where((x<12) &(8<x), 0.2 - 0.05*np.square(x-10),0)
    #height
    h = np.where( x >= 15, np.maximum(0, 0.1 - z + 0.1 * (25-x)/gravity),  np.maximum(0, 0.1 - z + 0.1 * (x-5)/gravity) )
    h = np.where( x < 5, np.maximum(0, 0.1 - z), h)
    #velocity
    u = 0*x
    v = np.where( x >= 15, -0.1/coriolis(x),  0.1/coriolis(x) )
    v = np.where( x < 5, 0, v)
    initial = np.stack((h,h*u,h*v,z))
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
                            gui={"nrows":2,"ncols":2,"num": 4})
