import numpy as np
#The stabilizer allows to run the hll-riemann solver in the case the eigenvalues of the Riemann problem are equal i.e. lambda_1=lambda_2 
#(dirty hack)
stabilizer = 1e-16 

class hll():
    """
    The hll class implements an approximate Riemannsolver using two 
    shock speeds.
    """
    def __init__(self, equation, eigenvalue_1, eigenvalue_2, 
                 converter=None, dry_tol = 1e-15):
        """ 
        Constructs a new instance of the hll class. 

        Parameters:
            equation (equation): Equation object implementing via 
                                 __call__(self, state)

            eigenvalue_1 (Callable): A function depending on state 
                                     variables and returning the first 
                                     eigenvalue

            eigenvalue_2 (Callable): A function depending on state 
                                     variables and returning the second
                                     eigenvalue

            converter (Callable):   Optional converstion function 
                                    primitive->conservative necessary
                                    if primitive values are provided

            dry_tol (float):        Minimum wet value > 0 (should be 
                                    near floating point accuracy)
        """
        self.equ = equation
        self.lambda_1 = eigenvalue_1 
        self.lambda_2 = eigenvalue_2 
        self.min_signal_speed = 0
        self.max_signal_speed = 0
        self.dry_tol = dry_tol
        if converter:
            self.converter = converter
        else:
            self.converter = lambda x: x    
        
    
    def calc_signal_speeds(self, left, right, mask_left = None, mask_right = None):
        """
        Compute the signal speeds of the Riemann problem using the provided (see __init__) eigenvalue methods 

        Parameters: 
            left (numpy.array):  The state values on the left side of the faces, 
                                 satisfying shape = (number of states, number of faces). 
            right (numpy.array): The state values on the left side of the faces, 
                                 satisfying shape = (number of states, number of faces). 
        Returns:
            numpy.array: Lower signal speed.
            numpy.array: Upper signal speed.
        """
        eigenvalues = np.stack((self.lambda_1(left) , self.lambda_2(left), self.lambda_1(right) , self.lambda_2(right)))
        if mask_left is not None and mask_right is not None:
            lower_signal_speed = np.zeros_like(left[0,:].squeeze())
            upper_signal_speed = np.zeros_like(left[0,:].squeeze())
            upper_signal_speed[mask_right] = np.max(eigenvalues[0:2,mask_right],axis = 0)
            lower_signal_speed[mask_right] = np.min(eigenvalues[0:2,mask_right],axis = 0)
            upper_signal_speed[mask_left] = np.max(eigenvalues[2:4,mask_left],axis = 0)
            lower_signal_speed[mask_left] = np.min(eigenvalues[2:4,mask_left],axis = 0)
            upper_signal_speed[np.logical_and(mask_left,mask_right)] = 0 
            lower_signal_speed[np.logical_and(mask_left,mask_right)] = 0 
        else:
            upper_signal_speed = np.max(eigenvalues,axis = 0)
            lower_signal_speed = np.min(eigenvalues,axis = 0)

        self.min_signal_speed = np.min(lower_signal_speed)
        self.max_signal_speed = np.max(upper_signal_speed)
        #print(np.maximum(np.abs(self.max_signal_speed),np.abs(self.min_signal_speed)))
        return lower_signal_speed, upper_signal_speed


    def __call__(self, left, right):
        """ Computes the solution to the Riemann problem. 

            Given the facial left and right state variables this method computes the 
            solution to the Riemann problem provided by the equation (see __init__)

            Parameters:
                left  (numpy.array): numpy array, with shape = (number of states, number of faces) 
                                    giving the left states at a/eacht face.
                right (numpy.array): numpy array, with shape = (number of states, number of faces) 
                                    giving the left states at a/eacht face.
            Returns:
                numpy.array: The flux function through a/each face 
        """
        idx_dry_left  = left[0] < self.dry_tol
        idx_dry_right = right[0] < self.dry_tol
        lower_signal_speed, upper_signal_speed = self.calc_signal_speeds(left,right)#,idx_dry_left ,idx_dry_right) 
        lower_signal_speed, upper_signal_speed = self.calc_signal_speeds(left,right) 

        idx_pos = 0 <= lower_signal_speed
        idx_neg = upper_signal_speed <= 0
        idx_int = np.logical_or(np.logical_not(idx_pos ),np.logical_not(idx_pos))

        idx_int_pos = np.logical_and(idx_int, idx_dry_right)
        idx_int_neg = np.logical_and(idx_int, idx_dry_left)
        idx_pos = np.logical_or(idx_pos,idx_int_pos)
        idx_neg = np.logical_or(idx_neg,idx_int_neg)
        idx_int = np.logical_or(np.logical_not(idx_pos ),np.logical_not(idx_pos))

        flux = np.zeros_like(left)
        loc_left = left[:,idx_pos]
        flux[:,idx_pos] = self.equ(loc_left)

        loc_right = right[:,idx_neg]
        flux[:,idx_neg] = self.equ(loc_right)

        loc_left   = left[:,idx_int]
        loc_right  = right[:,idx_int]
        sig1       = lower_signal_speed[idx_int] 
        sig2       = upper_signal_speed[idx_int] 
        flux[:,idx_int] =  (sig2 * self.equ(loc_left) - sig1 *self.equ(loc_right) + (sig1*sig2) *(self.converter(loc_right) - self.converter(loc_left)))/(sig2 - sig1 + stabilizer)
        return flux

