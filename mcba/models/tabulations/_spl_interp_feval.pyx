import numpy as np
cimport numpy as np

import cython
cimport cython

#
# See piecewise_cubic for comments
#

######################### Interpolation ###################################

cdef inline double spl_interp_feval(double x, double xa, double xb, double fa, 
                                    double fb, double fda, double fdb):
    cdef double dx = xb-xa
    cdef double dxa = x-xa
    cdef double dxb = xb-x

    cdef double deriv = (fa-fb)/dx
    cdef double A = -( fdb + deriv )
    cdef double B = fda + deriv

    cdef double f = A*dxa + B*dxb        
    f *= dxa*dxb/dx/dx
    f += fa*dxb/dx + fb*dxa/dx 
    return f
    


ctypedef np.int_t np_int_t

@cython.boundscheck(False)
@cython.wraparound(False)
def _spl_interp_feval_vect(np.ndarray[np_int_t] idx, 
                           np.ndarray[double] x_new, np.ndarray[double] xx, 
                           np.ndarray[double] yy, np.ndarray[double] yyd, 
                           np.ndarray[double] outp):

    cdef int i, j

    for i in range(idx.shape[0]):
        j = idx[i]
        outp[i] = spl_interp_feval(x_new[i], xx[j-1], xx[j],     # xa, xb
                                             yy[j-1], yy[j],    # fa, fb 
                                             yyd[j-1], yyd[j])  # fda, fdb

    return outp

   
