import numpy as np
import numerics.riemann as riemann
import numerics.reconstruct as reconstruct
import numerics.passive as passive
import functools
from numerics.temporal import euler
from numerics.utilities import to_conservative, to_primitive
from numerics.spatial import combine 
from boundary import do_nothing
import plotting

""" Unstable explicit method for the wave equation."""
class idx:
    h  = 0
    m  = 1
    size = 2
    h_m  = slice(0,2)

class solver:
    def __init__(self, grid, boundary, func_c):
        m_bc = boundary(1)
        h_bc = do_nothing(0)
        h_grid = boundary(1).grid(0.5*(grid[1:]+grid[0:-1])) # Use boundary class to extrapolate dual grid for nodes 
        h_rhs = staggered_rhs(h_grid,m_bc)
        m_rhs = staggered_rhs(grid,h_bc,func_c)
        rhs = combine([h_rhs, m_rhs],[idx.m, idx.h])
        self.integrator = euler(rhs)
        #self.integrator = euler(div)

    def __call__(self, states, t, dt):
       return self.integrator(states, t, dt, False)

class staggered_rhs:
    def __init__(self, grid, boundary, func_c = lambda x:1):
        #grid = boundary.grid(grid)
        self.func_c = func_c
        self.boundary = boundary
        self.dx = grid[1:] - grid[0:-1]

    def __call__(self, other_state, time):
        ext_state = self.boundary(other_state)
        c2 = np.square(self.func_c(time))
        return c2*(ext_state[0:-1] - ext_state[1:])/self.dx, 0 

class plotter(plotting.fvm):
    def pre_proc(self, state, time):
        left =  np.stack((state[0][0:-1],state[1]))
        right = np.stack((state[0][1:],state[1]))
        return left, right




    
 
    