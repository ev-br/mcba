"""
Off-diagonal matrix elements <fp | P | fq>.

Written by Matt.
"""
from __future__ import division, absolute_import, print_function
import numpy as np
from .ph_param import Lieb_Wu_a, initial_q

from .matrix_elements import deltas, bare_thetas, _Yfq2
from .partitions import buckets_from_pairs


# avoid extra lookups
_SUMM = np.add.reduce


def _preprocess_roots(roots, par):
    """Make sure "delta" & "buckets" keys are present in the the roots dict."""
    if "buckets" not in roots:
        fs_pairs = roots["partition"]
        roots["buckets"] = buckets_from_pairs(fs_pairs, par.N)
    roots["deltas"] = deltas(roots)
    roots["Yfq"] = np.sqrt(_Yfq2(roots, par))
    return roots


def _cached_bare_thetas(roots):
    """Return bare_thetas of element from deltas stored in cache."""
    return np.sin(-roots["deltas"])


def _thetas(roots, par):
    """Calculate thetas value of element from deltas stored in cache."""
    a = Lieb_Wu_a(par)
    return np.sqrt(a) * _cached_bare_thetas(roots)



#
# Calculate Y and Z matrices (eq (S57) (S58))
#
def _YZandK_mat(roots_p, roots, par):
    """
    Takes Bethe Ansatz roots of two eigenstates, returns Y and Z matrix
    between them (eq S57, S58)

    deltas come from eq (S37)
    thetas and sumth come from eq (S36)

    This function catches exceptional cases, but does not account for singular
    cases where c=+inf, this is as the implementation does not account for the
    difference in phase between the otherwise identical c=-inf and c=+inf
    states.

    This function assumes there is only one c=-inf configuration in the cache.
    The physics of the problem ensures this is the case.

    """

    # NOTE:
    #     This function is very ugly, but it's a very computationally expensive
    #     function, and separating it into different functions takes the
    #     performance down, so I'm leaving it ugly.

    # create local variables for use throughout
    assert roots_p['c'] != +np.inf and roots['c'] != +np.inf, \
                                            'Cannot use c=+inf states'
    N = par.N
    deltas_p, deltas_n = roots_p["deltas"], roots["deltas"]
    thetas_p, thetas_n = _thetas(roots_p, par), _thetas(roots, par)
    sumth_p, sumth = np.sum(_thetas(roots_p, par)), np.sum(_thetas(roots, par))
    root_p, root_n = roots_p["roots"], roots["roots"]

    # NONE of the below works if there is a c=+inf configuration in cache
    # NONE of the below works if there are TWO c=-inf configurations in cache
    # are assuming that there are not two singular configurations
    # i.e. no    pi*(-2,-1,0,1),  pi*(-1,0,1,2)
    # FAILS if there are two roots that happen to share a common value
    # e.g.   (1.23, 3.232 ,6.45,8.97)   and   (2.33,2.89, 3.232 ,7.43)

    # Create the K matrix using the X matrix (matrix of 2*i*(z_j - z_i)
    #
    # If clause handles the case of diagonal matrix elements (<f|P|f>)
    # For these matrix elements, the K matrix has a diagonal of 1's
    # Here two configs with shared roots would cause problems.
    # Assumes that no element of X is zero, there would be for shared root
    # Same reason would cause failure for two different c=-inf configs.
    #   the logic would put that case through incorrect function.
    #
    # Calculating the K and Z_K matrices accounts for the possibility that
    # we're given roots_p and roots that are the same.
    # I never actually do this in practice, as the value of the momentum
    # contribution is already calculated for those states, but the case is kept
    # in this function for the sake of making the function complete.

    # General case - Calculate K and Z_k matrices (eq (S59) for K
    if np.all(roots_p["partition"] != roots["partition"]):
        X = 2.j * np.subtract.outer(root_p, root_n)
        K_mat = (((np.exp(X) - 1.) / (X)) *  # Fail if roots_p, roots share val
                  np.exp(1.j * np.subtract.outer(-deltas_p, -deltas_n)))
        Z_K = root_p[:, np.newaxis] * K_mat

    # Exceptional cases - Calculate K and Z_K matrices
    else:
        if roots["c"] == -np.inf:  # K_mat == identity for singular diagonal
            K_mat = np.eye(N + 1)
            Z_K = root_p[:, np.newaxis] * K_mat
        else:
            # add 2pi*I to X - K_mat diagonal is zeros,
            # artificially add ones later
            X = 2.j*np.subtract.outer(root_p, root_n) + 2.j*np.pi*np.eye(N + 1)
            K_mat = (np.eye(N + 1) + ((np.exp(X) - 1.) / (X)) *
                     np.exp(1.j * np.subtract.outer(-deltas_p, -deltas_n)))
            Z_K = root_p[:, np.newaxis] * K_mat

    # Create Y and Z matrices, using K_mat and Z_k from above
    # No configuration singular => don't need to account for division by 1
    # One configuration singular => account for one value
    # Both configurations singular => account for both
    #
    # by 'account for' I mean manually put in relevant value for 0/0
    #
    #
    # If c=+inf have no clause to catch it.
    # but I only take the c=-inf values in my cache (same eigenstate)

    # General case - Calculate Y and Z matrices (eq (S57) (S58) resp.)
    if roots_p["c"] != -np.inf and roots["c"] != -np.inf:  # Neither singular

        Y = (K_mat -  # if c=-np.inf, fail.
             np.outer(np.sum(K_mat, axis=1), thetas_n) / sumth -
             np.outer(thetas_p, np.sum(K_mat, axis=0)) / sumth_p +
             np.outer(thetas_p, thetas_n) * np.sum(K_mat) /
                                        (sumth * sumth_p))[0:N, 0:N]


        Z = (2. / par.L) * (Z_K -
                            np.outer(np.sum(Z_K, axis=1), thetas_n) / sumth -
                            np.outer(thetas_p, np.sum(Z_K, axis=0)) / sumth_p +
                            np.outer(thetas_p, thetas_n) * np.sum(Z_K) /
                                          (sumth * sumth_p))[0:N, 0:N]


    # Exceptional cases - Calculate Y and Z matrices
    elif roots_p["c"] == -np.inf and roots["c"] == -np.inf:  # Both singular
        Y = (K_mat -
             np.sum(K_mat, axis=1) / (N + 1.) -
             np.sum(K_mat, axis=0) / (N + 1.) +
             np.sum(K_mat) / ((N + 1.)**2))[0:N, 0:N]

        Z = (2. / par.L) * (Z_K -
                            np.sum(Z_K, axis=1) / (N + 1.) -
                            np.sum(Z_K, axis=0) / (N + 1.) +
                            np.sum(Z_K) / ((N + 1.)**2))[0:N, 0:N]

    elif roots_p["c"] == -np.inf:  # primed is singular
        Y = (K_mat -
             np.outer(np.sum(K_mat, axis=1), thetas_n) / sumth -
             np.sum(K_mat, axis=0) / (N + 1) +
             thetas_n * np.sum(K_mat) / (sumth * (N + 1)))[0:N, 0:N]

        Z = (2. / par.L) * (Z_K -
                            np.outer(np.sum(Z_K, axis=1), thetas_n) / sumth -
                            np.sum(Z_K, axis=0) / (N + 1) +
                            thetas_n * np.sum(Z_K) / (sumth*(N + 1)))[0:N, 0:N]

    else:  # unprimed is singular
        Y = (K_mat -
             np.sum(K_mat, axis=1)[:, np.newaxis] / (N + 1) -
             np.outer(thetas_p, np.sum(K_mat, axis=0)) / sumth_p +
             thetas_p[:, np.newaxis] * np.sum(K_mat) /
                                        ((N + 1) * sumth_p))[0:N, 0:N]

        Z = (2. / par.L) * (Z_K -
                            np.sum(Z_K, axis=1)[:, np.newaxis] / (N + 1) -
                            np.outer(thetas_p, np.sum(Z_K, axis=0)) / sumth_p +
                            thetas_p[:, np.newaxis] * np.sum(Z_K) /
                                          ((N + 1) * sumth_p))[0:N, 0:N]

    return Y, Z


#-----------------------------------------------------------------------------#
#                             Calculating momentum                            #
#-----------------------------------------------------------------------------#

# Functions for single elements
def _take_row(length, val):
    """Return matrix of ones with one row 0, helper function for _deriv_where"""
    predmat = np.ones((length, length))
    predmat[val] = 0
    return predmat


def _deriv_where(roots_p, roots, par):
    """
    Return unnormalised momentum of sea for two different eigenstates.

    Uses line-replacement and the fact that are deriving at lambda==0
    det(A+B) == det(A)+
                sum(det(A with line x replaced by line x of B) +
                sum(det(A with line x,y replaced by line x,y of B) +
                ...
                det(B)


    Hence:
        d/dl(det(A+l*B)) at l==0 = sum(det(A with line x replaced by B)

    This function is a slower but more general version of _deriv, here as a
    backup in case the SVD does not converge.

    """
    A_mat, B_mat = _YZandK_mat(roots_p, roots, par)
    ret = sum(np.linalg.det(np.where(_take_row(par.N, i), A_mat, B_mat))
              for i in range(par.N))
    return ret


def _deriv(roots_p, roots, par):
    """
    Return unnormalised momentum of sea for two different eigenstates.

    This gives the expectation value of the momentum operator between two
    different eigenstates

    Uses SVD and relies on singular matrix - only works for roots_p and
    roots different.

    Calculates the derivative in Eqn (S56)

    """
    Y, Z = _YZandK_mat(roots_p, roots, par)
    # Y is real, remove rounding errors in complex part
    U, s, V = np.linalg.svd(Y.real)
    # inverse == transpose (matrices are orthogonal as Y is real)
    Uinv = U.T
    Vinv = V.T
    # is calculating the complex conjugate quicker?
    fac = np.linalg.det(U) * np.linalg.det(V)
    bottomrow = np.dot(Uinv[par.N - 1], Z.real)
    element = np.dot(bottomrow, Vinv[:, par.N - 1])

    return np.prod(s[:-1]) * element * fac


def fPf(roots_p, roots, par):
    r"""
    Matrix element <f | P | f>, non-trivial part of (S56).

    This is, in fact, <f_p | P_\down | f_q > : the momentum of the IMPURITY.

    Attempts to use SVD, if does not converge uses slower line-replacement
    method

    """
    Y_p = roots_p['Yfq']
    Y = roots['Yfq']

    # Normalisation in Eqn (S56) - needs to be applied for off-diagonal
    # elements, normalisation already applied for diagonal elements.
    matel_norm = 1. / (Y * Y_p)
    # If looking at diagonal element - use Eq(4.40) as SVD method doesn't work.
    #
    # If wanted, could add a filter here to only select state pairs that you
    # want to use e.g.
    #
    # if len(roots_p["partition"].p) > 1 or len(roots["partition"].p) > 1:
    #     return 0
    if roots_p["partition"] == roots["partition"]:
        matel = initial_q(par) - roots['P']
    else:
        try:
            matel = _deriv(roots_p, roots, par) * matel_norm
        # Just in case something doesn't work, print out problem states so can
        # investigate.
        except np.linalg.linalg.LinAlgError:
            # Print pairs and 'c' values of states where SVD doesn't converge
            print('\nLooking at this pair with the row replacement method')
            print(roots_p['partition'], '  ', roots_p['c'])
            print(roots['partition'], '  ', roots['c'])
            matel = _deriv_where(roots_p, roots, par) * matel_norm

    return matel



def _amp(roots_p, roots, par):
    """
    Return contribution to momentum of sea from roots_p and roots combination.

    Attempts to use SVD, if does not converge uses slower line-replacement
    method

    """
    FSfq_p = roots_p['FSfq']
    FSfq = roots['FSfq']

    matel = fPf(roots_p, roots, par)

    return FSfq * FSfq_p * matel


