"""
These are impurity-specific parts of the partitions: validity checks and 
a_partition.
"""
from __future__ import division, print_function, absolute_import

from .ph_param import Par
from ..partitions_base import * 

##################### Impurity-specific checks & validations #################
def sum_deltas(fs_pairs, par):
    """Just a shorthand:
    BA eq only has solutions when this is within [-(N+1), 0]  
    """
    return sum_FS(par.N) + sum(fs_pairs.p) - sum(fs_pairs.h) - par.m_q


def is_in_range(fs_pairs, par):
    """
    >>> par = Par(N=3, m_q=1)
    >>> fermi_sea(par.N)
    array([-2, -1,  0,  1])
    >>> is_in_range(fsPairs([], []), par), is_in_range(fsPairs([1], [2]), par) 
    (True, True)
    >>> is_in_range(fsPairs([-2], [2]), par)
    False

    >>> par = Par(N=4, m_q=1)
    >>> fermi_sea(par.N)
    array([-2, -1,  0,  1,  2])
    >>> is_in_range(fsPairs([2], [4]), par)
    False
    >>> is_in_range(fsPairs([1, 2], [-3, 3]), par)
    True
    """
    return -(par.N+1) <= sum_deltas(fs_pairs, par) <= 0 



def is_singular(fs_pairs, par):
    """A partition can be valid, but singular (i.e. c = \pm\inf )
    >>> par = Par(N=3, m_q=2)
    >>> fermi_sea(par.N)
    array([-2, -1,  0,  1])
    >>> is_in_range(fsPairs([],[]), par),  is_singular(fsPairs([], []), par)
    (True, True)
    >>> is_in_range(fsPairs([1],[2]), par),  is_singular(fsPairs([1],[2]), par)
    (True, False)
    
    >>> par = Par(N=4, m_q=1)
    >>> fermi_sea(par.N)
    array([-2, -1,  0,  1,  2])
    >>> fp = fsPairs([2], [3])
    >>> is_in_range(fp, par), is_singular(fp, par)
    (True, True)

    >>> fp = fsPairs([2], [4])
    >>> is_in_range(fp, par), is_singular(fp, par)
    (False, False)

    >>> fp = fsPairs([-2], [-3])
    >>> is_in_range(fp, par), is_singular(fp, par)
    (True, False)
    """
    momt = sum_deltas(fs_pairs, par)
    return momt == 0 or momt == -(par.N+1)




def is_valid(fs_pairs, par):
    return all(is_valid_h(_, par.N) for _ in fs_pairs.h) and \
           all(is_valid_p(_, par.N) for _ in fs_pairs.p) and \
           len(fs_pairs.p) == len(fs_pairs.h) and \
           len(fs_pairs.h) <=par.N +1 and \
           is_unique(fs_pairs) and is_in_range(fs_pairs, par)



################## Generating partitions ################################
"""This constructs a valid, non-singular partition.
To be used, e.g. as a starting point for the walkers.
"""

def _partition_gs(par):
    return fsPairs(h=[], p=[])

def _partition_mq(par):
    return fsPairs( h=[-sum_FS(par.N)-1], p=[par.m_q] )

def _partition_left(par):
    _L, _R = FL(par.N)
    return fsPairs( h=[_L], p=[_L-1] )


def a_partition(par):
    """Returns a valid, non-singular partition.

    >>> a_partition( Par(N=3, m_q=1) )
    fsPairs(h=[], p=[])
    >>> a_partition( Par(N=3, m_q=2) )
    fsPairs(h=[1], p=[2])
    >>> a_partition( Par(N=5, m_q=8) )
    fsPairs(h=[2], p=[8])
    >>> a_partition( Par(N=4, m_q=0) )
    fsPairs(h=[-2], p=[-3])
    """
    if par.m_q < 0:
        raise NotImplementedError("Q<0 not implemented.")
    
    _parts = [_partition_gs, _partition_mq, _partition_left]
    for c_pairs in _parts:
        pairs = c_pairs(par)
        if is_valid(pairs, par) and not is_singular(pairs, par): 
            return pairs

    # never get here
    raise RuntimeError("Failed to generate a_partition for %s" % par) 


##########################################################
if __name__ == "__main__":
    import doctest
    doctest.testmod()
