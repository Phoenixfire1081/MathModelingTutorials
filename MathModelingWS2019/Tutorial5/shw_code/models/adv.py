import numpy as np
import numerics.riemann as riemann
import numerics.reconstruct as reconstruct
import numerics.passive as passive
import functools
from numerics.temporal import heun_adaptive
from numerics.spatial import centered_muscl
from numerics.utilities import to_conservative, to_primitive


class idx:
    h  = 0
    hu = 1
    hv = 2
    size = 3
    h_hu      = slice(0,2)
    h_hu_hv   = slice(0,3)
    h_hu_hv_z = slice(0,4) 
    momentum  = slice(1,3)

class solver:
    def __init__(self, grid, boundary, equation):
        equ = equation
        reco = reconstruct.linear(idx.momentum)
        div = centered_muscl(grid, equ, boundary(2), reco) 
        self.integrator = heun_adaptive(div)
        #self.integrator = euler(div)

    def __call__(self, state, t, dt):
        return self.integrator(state, t, dt)


class equation():
    def __init__(self, func_m):
        self.func_m = func_m
        self.m = None 

    def update(self,time):
        self.m = self.func_m(time)
        

    def flux(self, primitive_l, primitive_r, source_jump):
        """Gives the left and the right hydrostatic reconstruction fluxes"""
        # hydrostatic correction for states
        flux_l = np.empty_like(primitive_l)
        flux_r = np.empty_like(primitive_r)
        flux_l[idx.h,:], flux_r[idx.h,:] = passive.flux(self.m,1,1)
        # passive transport for velocity 
        flux_l[idx.momentum,:], flux_r[idx.momentum,:] = passive.flux(self.m, primitive_l[idx.momentum,:], 
                                                                              primitive_r[idx.momentum,:])
        return flux_l, flux_r 

    def source_center(self, primitive_l, primitive_r, source_jump):
        """Gives the cell centered 2nd order correction source term."""
        return 0  

    
 
    