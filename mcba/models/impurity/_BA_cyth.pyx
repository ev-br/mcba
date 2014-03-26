import cython
cimport cython

################## A single BA eq ########################

cdef extern from "math.h":
  double tan( double )
  double sin( double )
  double M_PI
  

cpdef double BA_eq(double x, double c, int n, double a):
    return 1./tan(x) - a*(x+n*M_PI)+c
    
   

cpdef double dx_dc(double x, double a):
    """dx/dc for x a solution of BA_eq."""
    cdef double s
    s = sin(x)**2
    return s/(a*s + 1.)



cpdef Bethe_root_x0(double c, int n, double a):
    """Solution of a single BA eq for c\to-\infty (x\to 0)."""
    #
    # NB: when called from fsolve, c is an array, [c]
    #
    cdef double cb  = c - M_PI*a*n
    return -1./cb + (a+1./3)/cb**3


cpdef Bethe_root_xpi(double c, int n, double a):
    """Solution of single BA eq for c\to+\infty (x\to \pi)"""
    cdef double ch = c- M_PI*a*(n+1)
    return M_PI - 1./ch + (a+1./3)/ch**3   
