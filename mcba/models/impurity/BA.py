from __future__ import division, print_function, absolute_import

from scipy.optimize import brentq, fsolve 
import numpy as np

from .ph_param import Lieb_Wu_a
from .partitions import buckets_from_pairs, pairs_from_buckets, is_valid, sum_deltas
from ..tabulations.piecewise_cubic import PiecewiseCubic
#
# NB: more imports to be attempted at the very bottom 
# these will be the overrides of for the time-critical bits (BA_eq & splines).



##################### a single BA eq ################################
def BA_eq(x, c, n, a):
    """Originally, a Bethe root is defined on [pi*n, pi*(n+1)].
    Here, we subtract the pi*n, so that 0 < x < pi. 
    Incidentally, this alone delivers about 40% performance :-). 
    """
    return 1./np.tan(x) - a*(x+n*np.pi)+c

def dx_dc(x, a):
    """dx/dc for x a solution of BA_eq."""
    s = np.sin(x)**2
    return s/(a*s + 1.)




def Bethe_root_phase(c, n, a, xmin=1e-15, xmax=np.pi-1e-15):
    """For a Bethe root z=\pi*n + \delta, return 0< \delta <\pi."""
    x, res = brentq(BA_eq, xmin, xmax, args=(c, n, a), full_output=True)

    assert res.converged, \
        "\nBethe_root: brentq failed with c = %s, n = %s, a = %s " % (c, n, a)+\
        "conv = %s, iterations = %s" % (res.converged, res.iterations)

    return x


def Bethe_root_x0(c, n, a):
    """Solution of a single BA eq for c\to-\infty (x\to 0)."""
    #
    # NB: when called from fsolve, c is an array, [c]
    #
    cb = c - np.pi*a*n
    return  -1./cb + (a+1./3)/cb**3


def Bethe_root_xpi(c, n, a):
    """Solution of single BA eq for c\to+\infty (x\to \pi)"""
    ch = c- np.pi*a*(n+1)
    return np.pi - 1./ch + (a+1./3)/ch**3   



#######################################################################
####################### Solving a system of BA eq #####################
#######################################################################

class AbstractBASystemSolver(object):
    """An ABC for BA system solvers."""
    def __init__(self, par):
        self.par = par


    def solve(self, fs_pairs, startfrom=None):
        """Given fs_pairs, solve for the roots & c;
        If startfrom is not None, it's used as an initial guess. Either is None,
        or is a dict with "roots" and "c" as keys.
        Real work done in _solve. 
        NB: _solve receives buckets. It doesn't know about anything about fs_pairs.
        """
        assert is_valid(fs_pairs, self.par)
        buckets = buckets_from_pairs(fs_pairs, self.par.N)
        assert np.all(buckets == np.sort(buckets))

        momt = sum_deltas(fs_pairs, self.par)
        if momt == 0:
            c = -np.inf
            roots = np.pi*buckets
        elif momt == -(self.par.N + 1):
            c = np.inf
            roots = np.pi*(buckets + 1)
        else:
            c, roots = self._solve(buckets, startfrom)

        return {
            "buckets" : buckets,
            "roots" : roots,
            "c" :  c,
        }


    def find_roots(self, fs_pairs, c):
        """Given c, find the roots."""
        assert is_valid(fs_pairs, self.par)
        buckets = buckets_from_pairs(fs_pairs, self.par.N)

        if c == -np.inf:
            roots = np.pi*buckets
        elif c == np.inf:
            roots = np.pi*(buckets +1)
        else:
            roots = self._find_roots(buckets, c)

        return {
            "buckets" : buckets,
            "roots" : roots,
            "c" :  c,
        }


    def _solve(self, buckets, startfrom):
        raise NotImplementedError

    def _find_roots(self, buckets, c):
        raise NotImplementedError


################# the limit of \gamma\to\infty #############################
def c_lima0(buckets, par):
    """c in the a\to 0 limit. (all the phases are the same)"""
    return -1./np.tan( np.pi*(par.m_q - np.sum(buckets))/(par.N+1.) )


class BASolverInfGamma(AbstractBASystemSolver):
    """For \gamma = +\infty, there's not much to solve really.
    """
    def __init__(self, par):
        super(BASolverInfGamma, self).__init__(par)
        assert par.V == np.inf

    def _solve(self, buckets, startfrom):
        c = c_lima0(buckets, self.par)
        delta = np.pi*(self.par.m_q - np.sum(buckets))/(self.par.N+1)
        roots = np.pi*buckets + delta

        assert abs( np.sum(roots)-self.par.m_q*np.pi ) < 1e-10
        return c, roots




######################## Two-step solvers ###################################
class AbstractTwoStepSolver(AbstractBASystemSolver):
    """An ABC for the two-step solvers: Given c, solve N+1 1D equations for
    the roots (brentq), use the sum of roots as a lhs of the 1D equation for c.
    """
    def __init__(self, par):
        super(AbstractTwoStepSolver, self).__init__(par)
        # alternate starting guesses 
        self.start_points = [-0.1, 0.1, -1., 1.]


    def _solve(self, buckets, startfrom):
        """Set the starting point for the solver, solve the system, 
        check the result.
        """
        ### find the starting point for c
        if startfrom is not None: 
            c0 = startfrom["c"]
        else: 
            c0 = c_lima0(buckets, self.par)
        # if c0 is not zero, but is too close to it, fsolve fails to converge
        if abs(c0)<1e-10: 
            c0 = 0
        start_points = [c0] + self.start_points

        ### solve
        for s0 in start_points:
            cres, infd, ier, mesg = fsolve(self._Bethe_system_last,
                                       args=(buckets,), x0=s0, full_output=True)

            if ier != 1:         
                print ("""\n *** %s: fsolve failed w/ fs_pairs = %s,  c0 = %s 
                    full output: ier = %s, mesg = %s, infodict = %s, lhs() = %s
                    Falling back, trying a different starting pt...
                    """ % (self.__class__.__name__, pairs_from_buckets(buckets), 
                           s0, ier, mesg, infd, 
                           self._Bethe_system_last(infd['r'][0], buckets)) )
            else:
                # all ok, calc and return the roots
                roots = self._Bethe_system(cres, buckets)+ np.pi*buckets
                assert abs( self._Bethe_system_last(cres, buckets) ) <1e-9
                return cres[0], roots

        ### still not good, try falling back to an alternate solver
        cres, roots = self._fallback(buckets, startfrom)
        return cres[0], roots


    def _Bethe_system_last(self, c, buckets):
        """The last BA eq, to be fsolve-ed for c."""
        return np.add.reduce(self._Bethe_system(c, buckets)) \
                + np.pi*np.add.reduce(buckets) - np.pi*self.par.m_q


    def _Bethe_system(self, c, buckets):
        """The first N+1 of the BA equations: roots given c."""
        raise NotImplementedError


    def _find_roots(self, buckets, c):
        # NB: the order of arguments in _Bethe_system is tied by the solvers
        # (they want the first argument), while the callers use
        # buckets-then-c(or starting point for c) 
        return self._Bethe_system(c, buckets) + np.pi*buckets


    def _fallback(self, buckets, startfrom):
        """A fallback solver: somebody else's _solve."""
        raise NotImplementedError




class BASolverTwoStep(AbstractTwoStepSolver):
    """A two-step solver: Given c, solve N+1 1D equations for
    the roots, use the sum of roots as a lhs of the 1D equation for c.
    Use Brent's method for the first N+1 equations.
    """
    def __init__(self, par):
        super(BASolverTwoStep, self).__init__(par)
        self.a = Lieb_Wu_a(par)

    def _Bethe_system(self, c, buckets):
        """The first N+1 of the BA equations: roots given c."""
        t = np.asarray([Bethe_root_phase(c, n, self.a) for n in buckets])
        return t




class BASolverTwoStepTab(AbstractTwoStepSolver):
    """
    A two-step solver: Given c, solve N+1 1D equations for
    the roots, use the sum of roots as a lhs of the 1D equation for c.
    Instead of solving a single BA eq over and over, tabulate the 
    solution, x(c), and use that in the l.h.s. of the eq. for c.

    In case of a failure, fall back to Brent's method. 
    """
    def __init__(self, par, tab_base=1.02):   #base=1.02 in production
        super(BASolverTwoStepTab, self).__init__(par)
        self.tab_base = tab_base
        self.cc = self.make_grid(tab_base)
        a = Lieb_Wu_a(par)
        self.pa = a*np.pi

        self.tabulated = PiecewiseCubic(self.cc, 
                f=lambda c: Bethe_root_phase(c, 0, a), 
                fprime=lambda c: dx_dc( Bethe_root_phase(c, 0, a) , a),
                f_above=lambda c: Bethe_root_xpi(c, 0, a),
                f_below=lambda c: Bethe_root_x0(c, 0, a) 
        )
        
    def _Bethe_system(self, c, buckets):
        return self.tabulated(c - self.pa*buckets)


    def _fallback(self, buckets, startfrom):
        """In case of failure, fall back to the Brent's method."""
        # Can instantiate _solver @ first request, but is there a point: 
        # this function should only be invoked extremely rarely anyway. 
        _solver = BASolverTwoStep(self.par)
        c, roots = _solver._solve(buckets, startfrom)
        return (c,), roots


    def make_grid(self, base):
        """Cook up the grid for the tabulation of x(c)."""
        a = Lieb_Wu_a(self.par)

        # Tabulation range: find c large enough so that the c\to\pm\infty
        # limiting forms of the Bethe_root work well enough
        cmin = -20
        for _ in range(25):
            roots = [Bethe_root_phase(cmin, 0, a), Bethe_root_phase(-cmin, 0, a)]
            limits = [Bethe_root_x0(cmin, 0, a), Bethe_root_xpi(-cmin, 0, a)]
            matches = np.allclose(roots, limits, atol=1e-8, rtol=1e-8)
            if np.all(matches):
                break
            else:
                cmin *= 1.1

        if not np.all(matches):
            msg = "Solver's make_grid: cmin=%s is still not good enough." % cmin
            msg += "Stop."
            raise RuntimeError(msg)

        #proceed to tabulation
        xmax = np.pi/2.        
        xmin = -Bethe_root_phase(cmin, 0, a)
        Npts = int( np.log(abs(xmax/xmin)) / np.log(base) )

        xx0 = np.empty(Npts+1)
        for j in range(Npts+1):
            xx0[j] = xmax/base**(j+1)

        cc = np.array([-BA_eq(x, 0, 0, 0) for x in xx0[::-1]])
        cc = np.hstack((cc, np.array([0]), -cc[::-1]))

        return cc


# Override time-critical bits by a (way faster) Cython implementation
try:
    from ._BA_cyth import BA_eq, dx_dc, Bethe_root_x0, Bethe_root_xpi
except:
    pass


