r"""
Matrix elements and related functionality for the problem of a single impurity
injected into a sea of non-interacting fermions in one dimension.

Notation (mostly) follows
[0] H~Castella and X~Zotos, Phys Rev B \textbf{47}, 16186 (1993), and especially
[1] CJM.~Mathy, MB~Zvonarev, and E~Demler, Nature Physics \textbf{8}, 881 (2012).
Equation numbers, where given, correspond to either 
(i) the Supplementary Material of the latter paper, available at 
http://www.nature.com/nphys/journal/v8/n12/full/nphys2455.html
(ii) its' Feb 15, 2012 version

Technicalities:
* for N background particles, the number of roots is N+1.
* 'roots' here are the solutions of the BA equations (see BA.py). Specifically, 
  where roots is an argument, it's assumed to be a dict of the form
  { "buckets" : np.array of integers, length N+1, (n_j in MZ's notation) 
    "roots": np.array of floats, length N+1, (z_j in MZ's notation)
    "c" : float 
  }
* 'par' is a collection of N, L, V and m_q (see partitions)
* N is always par.N, the number of background fermions
* While in [1](i) the numbering is 1-based, here we are 0-based.
* Here we do not require N being even/odd: this is to be enforced by the models.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy as sc

from .ph_param import Lieb_Wu_a, initial_q
from .partitions import fermi_sea

# avoid extra lookups 
summ = np.add.reduce


#
# Basis states for N free fermions: are specified by an ordered set of N integers.  
#
def basis_FS(N):
    """Fermi sea of N particles.
    >>> basis_FS(3), basis_FS(5)
    (array([-1,  0,  1]), array([-2, -1,  0,  1,  2]))
    """
    return np.delete(fermi_sea(N), 0)

def basis_mq(N, m_q):
    """ A basis state with a single hole with momentum 2*pi*m_q/L. 
    >>> basis_mq(3, 1), basis_mq(3, 0)
    (array([-2, -1,  0]), array([-2, -1,  1]))
    >>> all(basis_mq(3, -2) == basis_FS(3))
    True
    """
    fs = fermi_sea(N)
    return np.delete( fs, (N+1)/2 + m_q  )


#
# Helpers for the roots: here the everything is in the impurity reference frame.
#
def energy(roots, par):
    """Energy of a state {z_j}."""
    return summ(roots["roots"]**2)*4./par.L**2


def deltas(roots, par):
    """ z_t = \pi*n_t + \delta_t
    MZ convention has an extra minus sign @ \delta.
    """
    return roots["roots"] - np.pi*roots["buckets"]


def bare_thetas(roots, par):
    """ \\theta_t = \sqrt{a} bare_theta_t, Eq. (S36)."""
    return np.sin( -deltas(roots, par) )  # MZ convention



def fq_Pup_fq(roots, par):
    r"""The non-trivial part of Eq. (4.40):
    the matrix element <f_q | P_\up | f_q> = q - ..., and 
    this function returns the ... from the rhs of (4.40).
    """
    assert len(roots["roots"]) == par.N +1
    assert np.all(np.sort(roots["buckets"]) == roots["buckets"])

    # For $c\to\infty$, all $\delta_t\to 0$, hence
    # the sums in (4.40) simplify to
    # \sum_t k_t / \sum_t 1 \equiv q / (N+1)
    if roots["c"] == -np.inf or roots["c"] == np.inf:
        return initial_q(par) / (par.N + 1)     # Matt's catch

    a = Lieb_Wu_a(par)
    bth2 = bare_thetas(roots, par)**2
    bth2e = bth2 / (1. + a* bth2)

    num = summ(bth2e*roots["roots"])
    denom = summ(bth2e)

    assert not np.allclose(denom, 0.), \
            "denom ==0 @ <f_q | P_up| f_q >"
    return (2./par.L)*num/denom


#
# Overlaps: < basis_state | f_q>
#

def _Yfq2(roots, par):
    """ The normalization factor: |Y_fq|^{-2}, Eq. (S48)."""
    if roots["c"] == -np.inf or roots["c"] == np.inf:
        return 1./(1.+par.N)

    a = Lieb_Wu_a(par)
    th = bare_thetas(roots, par)
    th2 = th**2
    e = 1. + a * th2
    Y = summ(th2/e)  * np.prod(e)
    Th = summ(th) 
    assert not np.allclose(Th, 0.), \
            "Th ==0 @ Yfq2"
    return Y / Th**2



def _detX(roots, basis_state, par):
    """ Det of an augmented Cauchy matrix,  Eq.(S53).
    NB: here basis_state is a set of *integers*.
    """
    assert par.N+1 == len(roots["roots"]) == len(basis_state) +1

    # Special cases, singular conf-s--- Eq.(S54)
    if roots["c"] == -np.inf or roots["c"] == np.inf:
        if np.all(basis_state == roots["buckets"][:-1]):
            return 1./(1.+par.N)
        elif np.all(basis_state == roots["buckets"][1:]):
            return -1./(1.+par.N)
        else:
            return 0.

    # Construct the matrix:
    th = bare_thetas(roots, par)
    Th = summ(th)
    assert not np.allclose(Th, 0.), \
            "Th ==0 @ _phi"
    u = np.pi*basis_state
    cauchy = 1./np.subtract.outer(u, roots["roots"])
    ph = -summ( (th/Th)*cauchy, axis=1)
   
    # NB: throwing away the last root
    X = cauchy[:, :-1]
    th1 = th[:-1]
    
    X += ph[:, np.newaxis]    # make ph a column vector  
    X *= th1
    det = sc.linalg.det(X, overwrite_a=True)

    """  # Above is equivalent to and is *way* faster then:
    X = np.empty([par.N,par.N])
    for l in range(par.N):
        X[l,:] = ph
        for j in range(par.N):
          X[l,j] +=  1./(self.u[j] - z[l])

    for l in range(par.N):
        X[l,:] *= bth[l]

    and a similar bit for the _phi.
    """
    return det



def overlap(roots, basis_state, par):
    """<basis state | f_q>. For basis_state=basis_FS, this is Eq. (4.45)/(S52).
    """
    assert np.all(np.sort(roots["buckets"]) == roots["buckets"])
    return _detX(roots, basis_state, par) / np.sqrt(_Yfq2(roots, par))



    
def det_Cauchy(roots, basis_state, par):
    """ Det of a Cauchy matrix,  Eq.(S53) w/out \phi.
    NB: here basis_state is a set of *integers*.
    NB: singular cnfs not taken care of --- just returns np.nan if Theta==0
    """
    assert par.N+1 == len(roots["roots"]) == len(basis_state) +1

    # Construct the matrix:
    th = bare_thetas(roots, par)
    Th = summ(th)
    if np.allclose(Th, 0.):
        # is probably a 'singular' state, just return NaN and pretend to be done
        return np.nan

    u = np.pi*basis_state
    cauchy = 1./np.subtract.outer(u, roots["roots"])
   
    # NB: throwing away the last root
    X = cauchy[:, :-1]
    th1 = th[:-1]
    X *= th1
    det = sc.linalg.det(X, overwrite_a=True)

    return det

