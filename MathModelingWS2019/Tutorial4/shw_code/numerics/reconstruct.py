import numpy as np
from abc import ABC, abstractmethod, abstractproperty

import numerics.limiter as limiter
from numerics.utilities import to_conservative, to_primitive 

class Reconstructor(ABC):
    def __init__(self, idx_conservative, pre_proc = lambda x:x, post_proc = lambda x:x):
        self.idx_conservative = idx_conservative
        self.pre_proc = pre_proc
        self.post_proc = post_proc
    def __call__(self,x,conservative):
        return self.state(x,conservative)

    @abstractmethod
    def state(self, grid, state):
        pass

    @abstractmethod
    def source(self, grid, state_face):
        pass


class constant(Reconstructor):
    def state(self,x,conservative):
        primitive = to_primitive( self.pre_proc(conservative), self.idx_conservative)
        # get rid of ghost cells
        primitive = primitive[:,1:-1]
        reco_l = primitive
        reco_r = primitive
        return self.post_proc(reco_l), self.post_proc(reco_r)
    def source(self, x,jump_at_face):
        reco_face = jump_at_face[:,1:-1]
        reco_cell = np.zeros(shape = (jump_at_face.shape[0],jump_at_face.shape[1]-3 ))
        return reco_face, reco_cell

class linear(Reconstructor):
    def state(self,x,conservative):
        primitive = to_primitive( self.pre_proc(conservative), self.idx_conservative)
        out_shape = list(primitive.shape)
        out_shape[1] -=2
        reco_l = np.empty(out_shape)
        reco_r = np.empty_like(reco_l)

        # ignore ghost cells
        dx = (x[2:-1] - x[1:-2])/2
        # compute slopes
        dxs = dx * limiter.minmod_slopes(x,primitive)
        # get rid of unecessary ghost cells
        primitive = primitive[:,1:-1]

        # variables in non conservative form 
        var = np.ones(out_shape[0],dtype=np.bool_)
        var[self.idx_conservative] = 0
        reco_r[var] = primitive[var] + dxs[var]
        reco_l[var] = primitive[var] - dxs[var]
    
        # all variables in conservative form (momentum) (check for zeros again ^^)
        var = self.idx_conservative 
        mask = primitive[0,:] < 1e-15
        reco_r[var,mask] = 0.0
        reco_l[var,mask] = 0.0
        mask = ~mask 
        reco_r[var,mask] =  primitive[var,mask] + reco_l[0,mask] / primitive[0,mask] *  dxs[var,mask]
        reco_l[var,mask] =  primitive[var,mask] - reco_r[0,mask] / primitive[0,mask] *  dxs[var,mask] 

        return self.post_proc(reco_l), self.post_proc(reco_r)
    def source(self,x,jump_at_face):
        dx_l = 0.5*(x[2:-1] - x[0:-3])
        dx_r = 0.5*(x[3:]   - x[1:-2])
        s = limiter.minmod(jump_at_face[:,0:-1]/dx_l,jump_at_face[:,1:]/dx_r)
        dx = (x[2:-1] - x[1:-2])
        reco_face = jump_at_face[:,1:-1] - 0.5*(dx[1:]* s[:,1:] + dx[0:-1] * s[:,0:-1] )
        reco_cell =  dx*s
        return reco_face, reco_cell

