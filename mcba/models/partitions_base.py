"""
The general bits for integer partitions: Fermi sea analogy and fsPairs, 
basic validity checks, buckets and back, serializations and deserializations.
This is to be extended by specific models.

The conventions are:
* 'buckets' is a np.array([-2, -1, 0, 1])
* 'pairs' is a namedtuple("fsPairs", "h p"), 
   so that fs_pairs.p/h is a list of particles/holes  

NB: 
 * buckets are sorted, particles and holes are not
 * is's always (holes, particles), in this order
 * fs_p is *fs_particles*, not *pairs* (fs_pairs are always spelled out)
 *  N is always par.N, so that len(buckets) == N+1

Ex: N=3, fermi sea is [-2, -1, 0, 1], so fs_h, fs_p == [], []
"""
from __future__ import division, print_function, absolute_import

from collections import namedtuple
import numpy as np


######################## Fermi sea analogy ###########################
def fermi_sea(N):
    """ 
    >>> fermi_sea(3)
    array([-2, -1,  0,  1])
    >>> fermi_sea(4)
    array([-2, -1,  0,  1,  2])
    """
    if N%2 == 1:
        return np.array(range(-(N+1)//2, (N+1)//2))
    else:
        return np.array(range(-N//2, N//2+1))


def FL(N):
    """Return the *inclusive* edges of the Fermi sea.
    >>> (FL(3)[0] == fermi_sea(3)[0]) and (FL(3)[1] == fermi_sea(3)[-1])
    True
    >>> FL(4)[0] == fermi_sea(4)[0] and FL(4)[1] == fermi_sea(4)[-1]
    True
    """
    if N%2 == 1:
        return -(N+1)//2, (N-1)//2
    else:
        return -N//2, N//2


def sum_FS(N):
    """ Sum of all buckets in the FS.
    >>> sum_FS(3), sum_FS(4)
    (-2, 0)
    """
    if N%2 == 1:
        return -(N+1)//2
    else:
        return 0

""" 
The main entity: the 'holes' under the Fermi level & 'particles' above it.
fsPairs.h and fsPairs.p are both lists.
"""
fsPairs = namedtuple("fsPairs", "h p")


######################### Checks #################################

def is_valid_h(h, N):
    """
    >>> N=3
    >>> fermi_sea(N)
    array([-2, -1,  0,  1])
    >>> [ is_valid_h(x, N) for x in [-2, -1, 0, 1, 42] ]
    [True, True, True, True, False]
    """
    return FL(N)[0] <= h <= FL(N)[1]


def is_valid_p(p, par):
    return not is_valid_h(p, par)



def is_unique(fs_pairs):
    """Neither holes not particles should not have duplicates.
    >>> is_unique(fsPairs([],[])), is_unique(fsPairs([1, 1], [5, 6]))
    (True, False)
    """
    return len(set(fs_pairs.h)) == len(fs_pairs.h) == \
           len(set(fs_pairs.p)) == len(fs_pairs.p)



##################### Buckets to pairs and back ########################
def pairs_from_buckets(buckets):
    """ Given buckets, return the pairs. NB: no validity checks made.
    
    >>> N = 3
    >>> pairs_from_buckets(np.array([-2, -1, 0, 1])) 
    fsPairs(h=[], p=[])
    >>> pairs_from_buckets(np.array([-2, -1, 0, 2]))
    fsPairs(h=[1], p=[2])
    
    >>> N = 4
    >>> pairs_from_buckets(np.array([-2, -1, 0, 1, 2]))
    fsPairs(h=[], p=[])
    >>> pairs_from_buckets(np.array([-2, -1, 0, 1, 3]))
    fsPairs(h=[2], p=[3])
    """
    assert buckets.dtype == 'int' and buckets.ndim == 1 and \
            all(np.sort(buckets) == buckets)
    
    fsea = fermi_sea(len(buckets) - 1)
    fs_particles = [x for x in buckets if x not in fsea]
    fs_holes = [x for x in fsea if x not in buckets]

    return fsPairs(fs_holes, fs_particles)



def buckets_from_pairs(fs_pairs, N):
    """ Given pairs, return the buckets. NB: need to explicitly give N.

    >>> N = 3
    >>> buckets_from_pairs(fsPairs([],[]), N) 
    array([-2, -1,  0,  1])
    >>> buckets_from_pairs(fsPairs([1], [2]), N) 
    array([-2, -1,  0,  2])

    >>> N=4
    >>> buckets_from_pairs(fsPairs([],[]), N)
    array([-2, -1,  0,  1,  2])
    >>> buckets_from_pairs(fsPairs([2], [3]), N)
    array([-2, -1,  0,  1,  3])
    """
    fsea = fermi_sea(N)
    assert all(h in fsea for h in fs_pairs.h)
    assert all(p not in fsea for p in fs_pairs.p)

    for h, p in zip(fs_pairs.h, fs_pairs.p):
        np.place(fsea, fsea==h, p)    
    fsea.sort()
    return fsea



################# Serializations ##################################
def pre_hash(fs_pairs):
    """Make fs_pairs hashable.
    NB: unless it's sorted, (2, 1) and (1, 2) would've been cached twice.
    >>> pre_hash(fsPairs([], [])), pre_hash(fsPairs([1, 2], [42, 4]))
    ((), (1, 2, 4, 42))
    """
    return tuple(sorted(fs_pairs.h)+sorted(fs_pairs.p))



def from_prehash(prehash):
    """Restore the pairs from its pre_hash value.
    >>> from_prehash(()),  from_prehash((1, 4)), from_prehash((-2, 1, -3, 5))
    (fsPairs(h=[], p=[]), fsPairs(h=[1], p=[4]), fsPairs(h=[-2, 1], p=[-3, 5]))
    """
    l = len(prehash)
    assert l%2 ==0
    return fsPairs(h=list(prehash[:l//2]), p=list(prehash[l//2:]))



##########################################################
if __name__=="__main__":
    import doctest
    doctest.testmod()

