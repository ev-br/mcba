r"""
Matrix elements of <P_imp> & <P_imp^2>, after O.G. (Dec 17 file version)
"""
import numpy as np
from numpy.linalg import det

from mcba.models.impurity import initial_q
from mcba.models.impurity.ph_param import Lieb_Wu_a


def dkdLa(dct, par):
    """OG Eq. (10).
    
    Input is  a line (df.loc[0]); output is the array for $dk_j / d\Lambda.
    """
    a = Lieb_Wu_a(par)
    L = par.L
    alpha = a * L / 2.
    kj = (2 / L) * dct["roots"]
    return (2 / L)  / ((alpha * kj - dct["c"])**2 + 1 + a) 



def D_loops(roots_k, roots_p, par):
    """D(k, p) factor.

    Using Eqs. (28), (51) and (34)
    `D_loops` and `D_gen` should give identical results (the latter is vectorized)
    """
    N = par.N
    
    dpdL = np.sqrt(dkdLa(roots_p, par))
    dkdL = np.sqrt(dkdLa(roots_k, par))
    
    m = np.empty((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            m[i, j] = dpdL[i] * dkdL[j]
            m[i, j] /= roots_p["roots"][i] - roots_k["roots"][j]
            m[i, j] *= 0.5 * par.L
            m[i, j] *= roots_p["c"] - roots_k["c"]
    return m


def D_gen(roots_k, roots_p, par):
    """Eq. (28) : D(k, p) for general k, p"""
    dpdL = np.sqrt(dkdLa(roots_p, par))       # TODO: can add to the df if slow
    dkdL = np.sqrt(dkdLa(roots_k, par))
    
    m = dpdL[:, None] * dkdL[None, :]
    m /=  (roots_p["roots"][:, None] - roots_k["roots"][None, :])
    m *= par.L / 2.
    m *= roots_p["c"] - roots_k["c"]
    
    return m


def D_inf(roots_p, roots_k, par):
    """Eq.(51): roots_k["c"] = -inf, roots_p is general."""
    assert roots_k["c"] == -np.inf
    assert roots_p["c"] != -np.inf
    
    N = par.N
    
    dpdL = np.sqrt(dkdLa(roots_p, par))

    m = np.empty((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            m[i, j] = dpdL[i]
            m[i, j] /= roots_p["roots"][i] - roots_k["roots"][j]   # sign: Eq (51) vs (28)
            m[i, j] /= np.sqrt(2/par.L)
    return m


def D(roots_k, roots_p, par):
    """D(k, p) matrix, all cases."""
    # Eq. (34)
    if roots_k["partition"] == roots_p["partition"]:
        return 1.0

    # Eq. (51)
    if not np.isfinite(roots_k["c"]):
        assert np.isfinite(roots_p["c"])
        return det( D_inf(roots_p, roots_k, par) )
    
    # transpose and retry
    if not np.isfinite(roots_p["c"]):
        assert np.isfinite(roots_k["c"])
        return D(roots_p, roots_k, par)

    # Eq. (28)
    return det(D_gen(roots_k, roots_p, par))



######### # $\langle p | P_{imp} | k \rangle$   ##########################

def pPk_gen(roots_p, roots_k, par):
    """Eq (44) : <p | P_imp | k>.
    
    This is for non-singular states, off-diagonal matrix elements.
    """
    
    N, L = par.N, par.L
    
    Dkp = D(roots_k, roots_p, par)  # note the order: follow O.G.'s argument order
    
    dkdL = dkdLa(roots_k, par)
    dpdL = dkdLa(roots_p, par)
    
    factor = Dkp / np.sqrt( dkdL.sum() * dpdL.sum())
    
    res = (roots_p["roots"]**2 - roots_k["roots"]**2).sum() / 2.0
    res *= (2 / L)**2
    
    res /= roots_p["c"] - roots_k["c"]
    
    return res * factor


def pPk_inf(roots_p, roots_k, par):
    """Eq (52) : < p | P_imp | k> for k being inf conf and p non-inf.
    """
    assert roots_k["c"] == -np.inf
    assert roots_p["c"] != -np.inf
    
    N, L = par.N, par.L

    Dkp = D(roots_k, roots_p, par)
    dpdL = dkdLa(roots_p, par)
    factor = Dkp / np.sqrt(dpdL.sum()) / np.sqrt(2.*(N+1)/L)
    
    res = ( roots_p["roots"]**2 - roots_k["roots"]**2 ).sum() / 2.    # sign change w.r.t. Eq. (52)
    res *= (2 / L)**2
    
    return res * factor


def S(n, dct, par):
    """OG Eq. (46)--(48)."""
    L = par.L
    k = dct["roots"] * (2 / L)
    return (k**n * dkdLa(dct, par)).sum()


def pPk_diag(roots, par):
    """Eqs. (46), (50)  : diagonal <k | P | k>"""
    if roots["c"] == -np.inf:
        return initial_q(par) / (par.N + 1)
    else:
        return S(1, roots, par) / S(0, roots, par)


def pPk(roots_p, roots_k, par):
    """<p | P_imp | k>, selector."""

    # Eq. (46), (50)
    if roots_p["partition"] == roots_k["partition"]:
        return pPk_diag(roots_p, par)
    
    # Eq. (52)
    if roots_k["c"] == -np.inf:
        assert roots_p["c"] != -np.inf
        return pPk_inf(roots_p, roots_k, par)
    
    # transpose and retry
    if roots_p["c"] == -np.inf:
        assert roots_k["c"] != -np.inf
        return pPk(roots_k, roots_p, par)
    
    # a general, off-diag case, Eq. (44)
    return pPk_gen(roots_p, roots_k, par)



################## # $\langle p | P_{imp}^2 | k \rangle$  #####################

def pP2k_gen(roots_p, roots_k, par):
    """Eq (45) : <p | P_imp^2 | k>.
    
    This is for non-singular states, off-diagonal matrix elements.
    """
    N, L = par.N, par.L
    a = Lieb_Wu_a(par)
    alpha = a * L / 2
    
    Dkp = D(roots_k, roots_p, par)  # note the order: follow O.G.'s argument order
    dkdL = dkdLa(roots_k, par)
    dpdL = dkdLa(roots_p, par)
    
    factor = Dkp / np.sqrt( dkdL.sum() * dpdL.sum())
    
    term = (roots_p["roots"]**3 - roots_k["roots"]**3).sum() / 3.0
    term *= (2 / L)**3
    term /= roots_p["c"] - roots_k["c"]
    
    term2 = (roots_p["roots"]**2 - roots_k["roots"]**2).sum() / 2.0
    term2 *= (2 / L)**2
    term2 /= roots_p["c"] - roots_k["c"]
    
    return factor * (term - alpha*term2**2)


def pP2k_diag(roots, par):
    """Eq (46) : <k | P_imp^2 | k>, diagonal matr elements."""
    if roots["c"] == -np.inf:
        return (roots["roots"]**2).sum() * (2/par.L)**2 / (par.N + 1)
    
    s0, s1, s2 = S(0, roots, par), S(1, roots, par), S(2, roots, par)
    
    a = Lieb_Wu_a(par)
    alpha = a * par.L / 2
    
    num = s2 + alpha*(s0*s2 - s1**2)
    return num / s0


def pP2k_inf(roots_p, roots_k, par):
    """Eq (53): <p | P_imp^2 | k> for k being inf conf and p general."""
    assert roots_k["c"] == -np.inf
    assert roots_p["c"] != -np.inf
    
    N, L = par.N, par.L

    Dkp = D(roots_k, roots_p, par)
    dpdL = dkdLa(roots_p, par)
    factor = Dkp / np.sqrt(dpdL.sum()) / np.sqrt(2.*(N+1)/L)
    
    res = ( roots_p["roots"]**3 - roots_k["roots"]**3 ).sum() / 3.    # sign change w.r.t. Eq. (53)
    res *= (2 / L)**3
    return factor * res


def pP2k(roots_p, roots_k, par):
    """<p | P_imp^2 | k>, selector."""
        
    # Eqs. (47), (50)
    if roots_p["partition"] == roots_k["partition"]:
        return pP2k_diag(roots_p, par)
    
    # Eq. (53)
    if roots_k["c"] == -np.inf:
        assert roots_p["c"] != -np.inf
        return pP2k_inf(roots_p, roots_k, par)
    
    # transpose and retry
    if roots_p["c"] == -np.inf:
        assert roots_k["c"] != -np.inf
        return pP2k(roots_k, roots_p, par)
    
    # a general, off-diag case, Eq. (45)
    return pP2k_gen(roots_p, roots_k, par)



