import numpy as np
import functools
from boundary import forcing_wall 
from models import shw


gravity  = 9.81
coriolis = lambda x: 0 
t_max    = 10
x_max    = 10 
num_grid = 301 
num_time = 21

plot_conservative = False

def time(n = num_time):
    return np.linspace(0, t_max, n)

def grid(n = num_grid):
    return np.linspace(0, x_max, n)

def initial(x):
    # orography
    z  = np.zeros_like(x) 
    # height
    h  = np.ones_like(x)
    # momenta
    hu = np.zeros_like(x)
    hv = np.zeros_like(x)
    initial = np.stack((h,hu,hv,z))
    # average using trapezoidal rule
    return 0.5*(initial[:,1:]+initial[:,0:-1])



boundary = forcing_wall
# bind gravity and coriolis force to shallow water solver constructor
# the resulting constructor will have signature solver(grid, boundary)
solver  = functools.partial(shw.solver, 
                            boundary = boundary,
                            equation = shw.equation(gravity, coriolis))
plotter = functools.partial(shw.plotter,
                            content = [[0,3],1,2], 
                            boundary = boundary(2), 
                            gui={"nrows":2,"ncols":2,"num": 4})