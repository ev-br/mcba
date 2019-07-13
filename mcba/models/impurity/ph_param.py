from __future__ import division, print_function, absolute_import

from math import pi
from collections import namedtuple

# http://ceasarjames.wordpress.com/2012/03/19/how-to-use-default-arguments-with-namedtuple/
class Par( namedtuple("Par", "N L V m_q") ):
    def __new__(cls, N=3, L=11, V=1., m_q=1):
        return super(Par, cls).__new__(cls, N, L, V, m_q)

# physical variables from microscopic parameters
def gamma(par):
    """Dimensionless coupling strength.  cf (3.5)"""
    return par.V / (1.*par.N/par.L)

def initial_q(par):
    """Momentum is quantized in 2*pi/L."""
    return par.m_q*2.*pi / par.L

def k_F(par):
    """Fermi momentum."""
    return pi*par.N/par.L

def E_F(par):
    """Fermi energy. """
    return k_F(par)**2
    
def E_FS(par):
    """Total energy of a filled Fermi sea."""
    return par.N * E_F(par) /3.

def E_in(par, mM=1.):
    r"""In-state energy: FS + injected impurity + \delta-function coupling.
    cf (2.12), (2.2)

    `mM` is the ratio of the host mass `m` to the impurity mass `M`: The kinetic
    energy of the impurity is
    $$
      \frac{p^2}{2M} = \frac{p^2}{2m} \frac{m}{M}
    $$
    and we use the units where $2m = 1$.
    """
    return E_FS(par) + (initial_q(par)**2 * mM) + 2.*par.V*par.N/par.L



def Lieb_Wu_a(par):
    """This enters the BA equations (cd BA.py & matrix_elements.py)"""
    return 4./(par.V* par.L)


def check_par(par):
    """N positive and less than L; L positive; V repulsive."""
    return par.N > 0 and par.L > 0 and par.N < par.L and par.V > 0


def par_id_str(par):
    """Is used by the DB.      
    >>> par_id_str(Par())
    'N3L11q1V1.00'
    """
    return "".join(["N{0}L{1}q{2}V{3:.2f}".format(par.N, par.L, 
                                                  par.m_q, par.V)])
    
def long_print(par):
    s = ["Par: N = {0},  L = {1},  ".format(par.N, par.L),
         "V = {0:.2f},  ".format(par.V),
         "gamma = {0:.2f}".format(gamma(par)),
         "\n",
         "     m_q = {0},  ".format(par.m_q),
         "q = {0:.2f},  ".format(initial_q(par)),
         "k_F = {0:.2f},  ".format(k_F(par)),
         "a = {0:.3f}".format(Lieb_Wu_a(par))
        ]
    return "".join(s)


##########################################################
if __name__ == "__main__":
    import doctest
    doctest.testmod()
