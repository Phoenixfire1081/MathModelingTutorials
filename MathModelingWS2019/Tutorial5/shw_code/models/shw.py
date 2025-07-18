import functools
import numpy as np
import numerics.riemann as riemann
import numerics.reconstruct as reconstruct
import numerics.passive as passive
from numerics.temporal import heun_adaptive
from numerics.spatial import centered_muscl
from numerics.utilities import to_conservative, to_primitive
import plotting
# The stabilizer allows to zero the gravity parameter as long as the coriolis frequency is zero too.
# (dirty hack)
stabilizer = 0  # 1e-16


class idx:
    """
    Helper (static) class to assign the field indices names and define selections (slices)
    """
    h = 0
    hu = 1
    hv = 2
    z = 3
    size = 4
    h_hu = slice(0, 2)
    h_hu_hv = slice(0, 3)
    h_hu_hv_z = slice(0, 4)
    momentum = slice(1, 3)


class solver:
    """ 
    This class provides time integration for a given grid, boundary
    conditions and a shallow water like equation.

    The integrator uses a CFL - adaptive Heun time integration scheme
    and is therefore of 2nd consistency order in time.
    The spatial discretization is realized by a conservative second 
    order finite volume method using  piecewise - linear (minmod) 
    hydrostatic reconstruction of h, u, v, h + z.  
    """

    def __init__(self, grid, boundary, equation):
        """ 
        Constructs an instance of the solver.

        Important: boundary needs to be constructor/class the object 
        itself. This ensures proper encapsulation, since the solver 
        has to pass the ghost size.

        Arguments:
            grid (numpy.array):        Coordinates of the nodes 

            boundary (Type(Boundary)): class of a boundary condition

            equation (Equation):       Instance of an shallow water 
                                       like equation  
        Returns:
            Object
        """
        equ = equation
        reco = reconstruct.linear(idx.momentum, hpz, hmz)
        bc = boundary(ghost_size=2)
        div = centered_muscl(grid, equ, bc, reco)
        self.integrator = heun_adaptive(div)
        #self.integrator = euler(div)

    def __call__(self, state, t, dt):
        """ 
        Invoke the solver at given time t to propagate the solution 
        to t + dt.

        Arguments:
            state (numpy.array): State at current time
            t (float):           Current time 
            dt (float):          Time inrecement 
        Returns:
            (numpy.array):       State at new time   
        """
        return self.integrator(state, t, dt)


def hpz(state):
    """ 
    Helper to reconstruct h + z
    """
    state[idx.z] += state[idx.h]
    return state


def hmz(state):
    """ 
    Helper to reconstruct h + z
    """
    state[idx.z] -= state[idx.h]
    return state


def hydrostatic_fix(state_l, state_r, source_jump):
    """
    Hydrostatic reconstruction introduced by Audusse Et al (2004)

    .. math::
        z^* = \\max(z_l,z_r) 
    .. math::
        h^*_l = h_l + z_l - z^*, h^*_r = h_r + z_r - z^*
    Implemented in the form of Bouchut (2007), where apparent 
    topography can be included by providing the jump over the face.

    Arguments: 
        state_l (numpy.array): States left of faces. 
                               shape = (number of fields, 
                                        number of faces)

        state_r (numpy.array): States right of faces. 
                               shape = (number of fields, 
                                        number of faces)

        source_jump (numpy.array): Jump of apparent_topography across 
                                   the face. shape = (number of fields,
                                                      number of faces)
    Returns:
        (numpy.array):  Hydrostatic well balance left and right states. 
    """
    fixed_l = np.copy(state_l)
    fixed_r = np.copy(state_r)
    dz = state_r[idx.z, :] - state_l[idx.z, :]
    fixed_l[idx.h, :] = np.maximum(state_l[idx.h, :]
                                   - np.maximum(dz + source_jump, 0), 0)
    fixed_r[idx.h, :] = np.maximum(state_r[idx.h, :]
                                   - np.maximum(-dz - source_jump, 0), 0)
    fixed_l[idx.z, :] = np.maximum(state_l[idx.z, :], state_r[idx.z, :])
    fixed_r[idx.z, :] = fixed_l[idx.z, :]
    return fixed_l, fixed_r


class equation():
    """
    Class defining the rotational shallow water equations with orography 
    and passive transversal velocity. 
    """

    def __init__(self, gravity, coriolis):
        """ Constructs an object of eqaution"""
        
        #        self.dx = (x[2:-1] - x[1:-2])
        self.gravity = gravity
        self.coriolis = coriolis
        self.max_signal_speed = 0
        # partially bind converter for riemann solver.
        # Riemann problem is only solved in 1D!
        p_to_conservative = functools.partial(to_conservative, idx=idx.hu)
        self.solve_riemann = riemann.hll(
            self.equ, self.lower_eigenvalue, self.upper_eigenvalue, p_to_conservative)

    def update(self, time):
        """ Handles time dependence (does nothing here) """
        pass

    def source_jump_at_face(self, x, conservative):
        source_vorticity = self.coriolis(x[1:-1]) * (x[2:] - x[0:-2])*0.5
        primitive = to_primitive(conservative, idx.momentum)
        return np.stack((-source_vorticity/(self.gravity + stabilizer)*0.5*(primitive[2, 0:-1]+primitive[2, 1:]), source_vorticity))

    def equ(self, primitive):
        """ 
        Evaluates the analytical fluxes for the classical 1d shallow 
        water equations.
        Arguments:
            primitive (numpy.array): Flux function argument of shape = 
                                     (number of fields, number of faces) 
        
        Returns:
            (numpy.array): Evaluated flux.
        """
        eq = np.empty_like(primitive)
        eq[idx.h] = primitive[idx.h, :] * primitive[idx.hu]
        eq[idx.hu] = primitive[idx.h, :] * \
            np.square(primitive[idx.hu, :]) + \
            self.pressure(primitive[idx.h, :])
        return eq

    def pressure(self, density):
        """ Evaluates the hydrostatic pressure for given density field."""
        return 0.5 * self.gravity * np.square(density)

    def lower_eigenvalue(self, primitive):
        """
        Evaluates the lower Eigenvalue of the Riemann problem for given 
        primitive state. 

        Arguments:
            primitive (numpy.array): States in primitive form.
        Returns:
            (numpy.array): The lower Eigenvalue for the Shallow Water 
                           Riemann problem. 
        """
        return primitive[idx.hu, :] - np.sqrt(self.gravity * primitive[idx.h, :])

    def upper_eigenvalue(self, primitive):
        """
        Evaluates the upper Eigenvalue of the Riemann problem for given 
        primitive state. 

        Arguments:
            primitive (numpy.array): States in primitive form.
        Returns:
            (numpy.array): The upper Eigenvalue for the Shallow Water 
                           Riemann problem. 
        """
        return primitive[idx.hu, :] + np.sqrt(self.gravity * primitive[idx.h, :])

    def flux(self, primitive_l, primitive_r, source_jump):
        """
        Computes the 1.5D numerical Shallow Water flux at the interfaces.

        Arguments:
            primitive_l (numpy.array): States at the left side of the face in primitive form.

            primitive_r (numpy.array): States at the right side of the face in primitive form.

        Returns:
            (numpy.array): 
        """ 
        # hydrostatic correction for states
        fixed_l, fixed_r = hydrostatic_fix(
            primitive_l, primitive_r, source_jump[0, :])
        # symmetric flux
        symmetric = self.solve_riemann(
            fixed_l[idx.h_hu, :], fixed_r[idx.h_hu, :])
        self.max_signal_speed = np.maximum(np.abs(
            self.solve_riemann.max_signal_speed), np.abs(self.solve_riemann.min_signal_speed))
        flux_l = np.empty_like(primitive_l)
        flux_r = np.empty_like(primitive_r)
        flux_l[idx.h_hu, :] = symmetric
        flux_r[idx.h_hu, :] = symmetric
        # add hydrostatic correction for fluxes
        flux_l[idx.hu, :] += (self.pressure(primitive_l[idx.h, :]
                                            ) - self.pressure(fixed_l[idx.h, :]))
        flux_r[idx.hu, :] += (self.pressure(primitive_r[idx.h, :]
                                            ) - self.pressure(fixed_r[idx.h, :]))
        # passive transport for transversal velocity
        flux_l[idx.hv, :], flux_r[idx.hv, :] = passive.flux(symmetric[idx.h, :],
                                                            fixed_l[idx.hv, :],
                                                            fixed_r[idx.hv, :],
                                                            source_jump[1, :])
        # orography is not changing
        flux_l[idx.z, :] = 0
        flux_r[idx.z, :] = 0
        return flux_l, flux_r

    def source_center(self, primitive_l, primitive_r, source_jump):
        """
        Gives the cell centered 2nd order correction source term.
        Arguments:
            primitive_l (numpy.array): States at the left side of the 
                                       face in primitive form.
            primitive_r (numpy.array): States at the right side of the 
                                       face in primitive form.
        Returns:
            (numpy.array):
        """
        flux = np.empty_like(primitive_l)
        flux[idx.h, :] = 0.0
        flux[idx.hu, :] = -0.5 * self.gravity * (primitive_l[idx.h, :] + primitive_r[idx.h, :]) \
            * (primitive_r[idx.z, :] - primitive_l[idx.z, :] + source_jump[0, :])
        flux[idx.hv, :] = -0.5 * (primitive_l[idx.h, :] * primitive_l[idx.hu, :]
                                  + primitive_r[idx.h, :] * primitive_r[idx.hu, :]) * source_jump[1, :]
        flux[idx.z, :] = 0.0
        return flux


class plotter(plotting.fvm):
    """ A Shallow Water spezialized plotter derived from finite volume plotter. """
    def init(self, **kwargs):
        if "boundary" in kwargs:
            self.boundary = kwargs["boundary"]
            self.ext_grid = self.boundary.grid(self.grid)
        else:
            raise TypeError("Missing necessary boundary condition object.")

        if "conservative" in kwargs:
            self.conservative = kwargs["conservative"]
        else:
            self.conservative = False

        self.reconstruct = reconstruct.linear(idx.momentum, hpz, hmz)

    def pre_proc(self, state, time):
        ext_state = self.boundary(state, time=time)
        left, right = self.reconstruct(self.ext_grid, ext_state)
        fleft, fright = hydrostatic_fix(left, right, 0)
        fleft[idx.h] += fleft[idx.z]
        fright[idx.h] += fright[idx.z]
        if not self.conservative:
            fleft[idx.momentum] /= fleft[idx.h]
            fright[idx.momentum] /= fright[idx.h]
        return fleft[:, 1:-1], fright[:, 1:-1]
