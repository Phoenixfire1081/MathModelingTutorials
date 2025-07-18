import numpy as np
def to_primitive(conservative, idx,  dry_tol=1e-14):
    primitive = np.copy(conservative)
    mask = conservative[0,:] < dry_tol
    primitive[idx, mask] = 0.0
    mask = ~mask
    primitive[idx, mask] = conservative[idx,mask]/conservative[0,mask]
    return primitive

def to_conservative(primitive, idx):
    conservative = np.copy(primitive)
    conservative[idx,:] = primitive[idx,:]*primitive[0,:]
    return conservative 
