import numpy as np

class Boundary:
    """
    Base class for boundary conditions.
    
    This class provide basic functionality needed by derived 
    boundary conditions.
    """
    def __init__(self, ghost_size, extend = True):
        self.ghost_size = ghost_size
        self.pad_shape = ((0,0),(self.ghost_size,self.ghost_size))
        self.inner_slice = slice(ghost_size,-ghost_size)
        if extend == True:
            self.view_slice = slice(None)
        else:
            self.view_slice = self.inner_slice 
    
    def view(self,x):
            return np.atleast_2d(x)[:,self.view_slice]

    def inner(self,x):
        return np.atleast_2d(x)[:,self.inner_slice]

class do_nothing(Boundary):
    def __call__(self, state, time = 0):
        return state
    def grid(self,grid):
        return grid
   
class neumann(Boundary):
    def __call__(self, state, time = 0):
#        pdb.set_trace()
        return np.pad(self.view(state),self.pad_shape,'symmetric').squeeze()
    def grid(self, grid):
        return np.pad(self.view(grid).squeeze(), (self.ghost_size, self.ghost_size),'reflect', reflect_type='odd')
    
class periodic(Boundary):
    def __call__(self, state, time = 0):
        return np.pad(self.view(state),self.pad_shape,'wrap').squeeze()
    def grid(self, grid):
        x = self.view(grid).squeeze()
        right = x[-1] - x[0] + x[1:self.ghost_size+1]
        left =  x[0] - x[-1] + x[-self.ghost_size-1:-1]
        return np.concatenate((left,x,right))

class wall(neumann):
    def __call__(self, state, time = 0, momentum_index = 1):
        tmp = np.pad(self.view(state),self.pad_shape,'symmetric').squeeze()
        tmp[ momentum_index,0:self.ghost_size] *= (-1)
        return tmp

class bouchut_3_BC(Boundary):
    def __call__(self, state, time):
        tmp = np.pad(self.view(state),self.pad_shape,'symmetric').squeeze()
        tmp[0][-self.ghost_size:] = 0.33
        tmp[1][:self.ghost_size] = 0.18
        tmp[2][:self.ghost_size] = 0
        return tmp
    def grid(self, grid):
        return np.pad(self.view(grid).squeeze(),(self.ghost_size,self.ghost_size),'reflect', reflect_type='odd')

class right_wall(Boundary):
    h0 = 0.5
    delta = 0.1
    omega = 10
    def __call__(self, state, time):
        tmp = np.pad(self.view(state),self.pad_shape,'symmetric').squeeze()
        tmp[0][-self.ghost_size:] = 0.0	# rigid wall
        tmp[0][:self.ghost_size] = self.f(time)
        return tmp
        
    def grid(self, grid):
        return np.pad(self.view(grid).squeeze(),(self.ghost_size,self.ghost_size),'reflect', reflect_type='odd')

    def f(self,t):
        return self.h0 + self.delta * (1-np.cos( self.omega * t))
	
    
class forcing_wall(Boundary):
    frequency = 2*np.pi/1000
    amplitude = 0.001 
    mean = 1.0
    def __call__(self, state, time):
        tmp = np.pad(self.view(state),((0,0),(0, self.ghost_size)),'symmetric')
        tmp[1,-2:] *= (-1)
        left_inner = self.view(state)[:,0]
        h_left  = 2 * self.f(time)  -  left_inner[0]
        #hu_left = left_inner[1] + (self.inner_x[1] - self.inner_x[0]) * self.df(time)
        tmp = np.pad(tmp[:,:],((0,0),(self.ghost_size, 0)),'symmetric' ).squeeze()
        tmp[0 , 0:2] = h_left
        #tmp[1 , 0:2] = hu_left 
        tmp[2:, 0:2] = tmp[2:,2]  
        return tmp

    def grid(self, grid):
        return np.pad(self.view(grid).squeeze(),(self.ghost_size,self.ghost_size),'reflect', reflect_type='odd')

    def f(self,t):
        return self.mean + self.amplitude * (1-np.cos( self.frequency * t))
    def df(self,t):
        return  self.frequency*self.amplitude * np.cos(self.frequency * t)

class forcing_wall_ray(forcing_wall):
    def __init__(self, ghost_size, amplitude, frequency, mean):
        self.mean = mean
        self.amplitude = amplitude
        self.frequency = frequency
        super(forcing_wall_ray, self).__init__(ghost_size = ghost_size)

def grid_periodic(x,ghost_size):
    right = x[-1] - x[0] + x[1:ghost_size+1]
    left =  x[0] - x[-1] + x[-ghost_size-1:-1]
    return np.concatenate((left,x,right))

def grid_mirror(x,ghost_size):
    return np.pad(x,(ghost_size,ghost_size),'reflect', reflect_type='odd')
