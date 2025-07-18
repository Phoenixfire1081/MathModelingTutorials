import numpy as np
import functools 
from boundary import periodic
from models import shw


gravity  = 9.8
coriolis = lambda x: 0
t_max    = 0.2
x_max    = 1 
num_time = 100
num_grid = 41
plot_conservative = False
def time(n = num_time):
    return np.linspace(0, t_max, n)

def grid(n = num_grid):
    return np.linspace(0, x_max, n)

def initial(x):
    #orography
    z = 0 * np.ones_like(x)
    #height
    h = 2 + np.sin(2 * np.pi * x)
    #velocity
    u = 0 * np.ones_like(x)
    v = 0 * np.ones_like(x)
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
