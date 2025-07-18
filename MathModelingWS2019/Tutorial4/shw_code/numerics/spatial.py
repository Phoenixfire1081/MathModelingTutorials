import numpy as np
import abc

class centered_muscl:
    """ 
    Class implementing a finite volume right hand side for hyperbolic 
    first order pdes with source term.
    
    """
    def __init__(self, grid, equation, boundary, reconstructor):
        """
        Constructs an instance of centered_muscl

        Arguments:
        grid (numpy.array): Array holding the nodal grid points

        equation (Equation): The object providing update, flux, 
                             source_jump_at_face methods

        boundary (Boundary): An object specifying the boundary condition 
                             derived from Boundary 
        
        reconstrutor (Reconstructor): An object for the reconstruction 
                                      of the facial values from the 

        Returns:
            (Object): The new instance.
        """

        self.equation = equation 
        self.reconstruct = reconstructor
        self.boundary = boundary
        self.x  = boundary.grid(grid) 
        self.dx = boundary.inner((self.x[1:]-self.x[0:-1]))
        self.min_dx = np.min(self.dx) 


    def __call__(self,state, time = 0):
        """ 
        Evaluates the rhs at states and given point in time.
        """
        ext_state = self.boundary(state, time = time)
        left, right = self.reconstruct(self.x, ext_state)
        self.equation.update(time)
        source_jumps = self.equation.source_jump_at_face(self.x, ext_state)
        source_face, source_cell = self.reconstruct.source(self.x,source_jumps)
        flux_right, flux_left = self.equation.flux(right[:,0:-1],left[:,1:], source_face) 
        flux_center = self.equation.source_center(left, right, source_cell)
        dt_max = 0.5*self.min_dx/self.equation.max_signal_speed
        return (flux_left[:,0:-1] - flux_right[:,1:] + flux_center[:,1:-1])/self.dx, dt_max
    

class combine:
    """ Class that collects different right hand side operators. """

    def __init__(self, operators, indices ):
        """
        Constructs combine object
        
        Arguments:
            operators (List): List of rhs operators to evaluate.

            indices (List):   List of state indices to pass to each 
                              operator

        Returns:
            (Object)
        """

        self.operators = operators
        self.dt_max = 0
        self.indices = indices
    
    def __call__(self, states, time = 0):
        results = []
        overall_dt_max  = np.inf
        for (operator, idx) in zip(self.operators, self.indices):
            result, dt_max = operator(states[idx], time)
            overall_dt_max = min(dt_max, overall_dt_max)
            results.append(result)
        return np.array(results), overall_dt_max            

