import numpy as np
#def minmod(a,b):
#    result = np.zeros_like(a)
#    idx_pos = np.logical_and( 0 <= a, 0 <= b)
#    idx_neg = np.logical_and( 0 >= a, 0 >= b)
#    result[idx_pos] = np.minimum(a[idx_pos],b[idx_pos])
#    result[idx_neg] = np.maximum(a[idx_neg],b[idx_neg])
#    return result

#
#           y_0        y_1        y_2 
#     
#     |   -----   |  ------  |  -----  |
#    x_0         x_1        x_2       x_3
def minmod_slopes(x,y):
    return slopes_diff(x,y[:,1:-1] - y[:,0:-2], y[:,2:] - y[:,1:-1])

def minmod(a,b):
    return np.maximum(0,np.minimum(a,b)) + np.minimum(0,np.maximum(a,b))

def slopes_diff(x,dz_l,dz_r):
    dx_l = 0.5*(  x[2:-1] - x[0:-3])
    dx_r = 0.5*(  x[3:] - x[1:-2]  )
    return minmod(dz_l/dx_l, dz_r/dx_r)
 


