import numpy as np

def flux(advection, tracer_l, tracer_r, source_jump = 0):
    """
    Passive transport realized by upwind flux with additional source.

    Arguments:
        advection (numpy.array): Advection velocity at the face
        
        tracer_l  (numpy.array): State which will be advected, on the 
                                 left side of the face.

        tracer_r  (numpy.array): State which will be advected, on the 
                                 right side of the face.
    Return:
       (numpy.array): Classical upwind flux, with optional additional 
                      source term.
    """
    a_pos = np.maximum(advection,0)
    a_neg = np.minimum(advection,0)
    flux_l = a_pos * tracer_l + a_neg * (tracer_r + source_jump)
    flux_r = a_pos *(tracer_l - source_jump) + a_neg * (tracer_r)
    return flux_l, flux_r 

