import numpy as np
import functools 
from boundary import neumann
from models import shw


gravity  = 9.81
coriolis = lambda x: 2*np.pi/50
t_max    = 200
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
    h = 0.33 * np.ones_like(x)
    #velocity
    u = 0.18 / h
    v = 0 * x
    initial = np.stack((h,h*u,h*v,z))
    # average using trapezoidal rule
    return 0.5*(initial[:,1:]+initial[:,0:-1])

boundary = bouchut_3_BC

# bind gravity and coriolis force to shallow water solver constructor
# the resulting constructor will have signature solver(grid,boundary)
solver  = functools.partial(shw.solver, 
                            boundary = boundary,
                            equation = shw.equation(gravity, coriolis))
plotter = functools.partial(shw.plotter,
                            content = [[0,3],1,2], 
                            boundary = boundary(2), 
                            gui={"nrows":2,"ncols":2,"num": 4})
