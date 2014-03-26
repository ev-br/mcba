from __future__ import division, print_function, absolute_import

import numpy as np
#
# NB: _spl_interp_feval_vect to be overridden by a Cython implementation,  
# if available
#

def _spl_interp_feval_vect(idx, x_new, xx, ff, ffd, buf):
    """Piecewise cubic interpolation helper.
    
    Parameters:
    xx, ff, ffd:  1D arrays.  tabulated values (arg, f, f')
    x_new: 1D array. evaluate the function at these values
    idx: 1D array. indices *in the tabulated arrays* of the corresponding x_news
         Sizes of idx and x_new must agree
    buf: 1D array. A preallocated buffer, size at least the size of idx.
         NB: not used here, but *is* used in cython implementation.
    """
    idx1 = idx-1
    
    xa, xb = xx[idx1], xx[idx]
    fa, fb = ff[idx1], ff[idx]
    fda, fdb = ffd[idx1], ffd[idx]

    dx = xb-xa
    dxa, dxb = (x_new-xa)/dx, (xb-x_new)/dx

    f = -(fdb*dx + fa-fb)*dxa + (fda*dx + fa-fb)*dxb
    f *= dxa*dxb
    f += fa*dxb + fb*dxa

    return f


class PiecewiseCubic(object):
    """Piecewise cubic interpolator."""

    def __init__(self, xx, f, fprime, f_above, f_below):
        """
        xx: array_like, 1d. grid values
        f: a callable, f(xx) the values to tabulate
        fprime: a callable, derivative of f(x)
        f_above: a callable, used for f(x > max(xx))
        f_below: a callable, used for f(x < min(xx))
        """
        self.xx = np.atleast_1d(xx)
        self.yy = np.array([f(x) for x in self.xx])
        self.yyd = np.array([fprime(x) for x in self.xx])
        self.n_tab = len(self.xx)

        self.f_above = np.vectorize(f_above)
        self.f_below = np.vectorize(f_below)

        #preallocate the buffer: shaves off a wee bit of exec time
        self.buf = np.empty(self.n_tab)

    def __call__(self, x_new):
        """User-callable: interpolate for the values of x_new within bounds, 
           use f_above/below for the values outside.
        """
        x_new = np.atleast_1d( x_new )
        below_range = x_new < self.xx[0]
        above_range = x_new > self.xx[self.n_tab-1]
        
        idx = np.searchsorted(self.xx, x_new) 
        idx = np.clip(idx, 1, self.n_tab-1, out=idx)

        if len(x_new)> len(self.buf):
            self.buf = np.empty(len(x_new))

        res = _spl_interp_feval_vect(idx, x_new, 
                self.xx, self.yy, self.yyd, self.buf)[0:len(x_new)]

        if np.any(below_range):
            res[below_range] = self.f_below(x_new[below_range])
        if np.any(above_range):
            res[above_range] = self.f_above(x_new[above_range])

        return res


try:
    from ._spl_interp_feval import _spl_interp_feval_vect
except:
    pass
        
