import numpy as np
from boundary import neumann as boundary 
import pdb
class euler:
    def __init__(self, rhs):
        self.rhs = rhs 
    def __call__(self, state, t, dt, check_cfl = True):
        rhs, dt_max = self.rhs(state, t)
        if check_cfl:
            #print("current time evaluation at %.5f, max stepsize = %.5f"%(t_old,dt_max))
            dt = min(dt,dt_max)
        return state +  dt * rhs, dt

class euler_adaptive:
    def __init__(self, rhs):
        self.rhs = rhs 
    def __call__(self, state , t, dt):
        current = state 
        t_current = t
        t_end = t + dt
        while t_current < t_end:
            rhs, dt_max = self.rhs(current,t_current)
            dt = min(t_end - t_current, dt_max)
            current = current + dt * rhs
            print(dt)
            t_current += dt
        return current, t_current-t 

class heun_adaptive:
    def __init__(self, rhs):
        self.rhs = rhs
        self.euler = euler(rhs)
    def __call__(self, state , t, dt):
        current = state 
        pred= current 
        t_current = t
        t_end = t + dt
        dt_wish = dt
        while t_current < t_end:
            while True:
                pred, dt_1 = self.euler(current, t_current, dt_wish)
                pred, dt_2 = self.euler(pred, t_current + dt_1, dt_1)
                if dt_2 + 1e-15 < dt_1:
                   dt_wish = dt_2 
                else:
                    t_current += dt_2
                    dt_wish = np.minimum(dt,t+dt-t_current)
                    current = 0.5*(current + pred)
                    break
            #print("Current time: %f"%t_current)
        return current, t_current-t 
    
