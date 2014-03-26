from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.stats import planck

from .models.partitions_base import FL, fsPairs


#################### helpers ###########################################
def randomchoice(lst):
    """random.choice(), essentially. (why is it not implemented in numpy.random)"""
    return lst[ np.random.randint(len(lst)) ]    



def patch_list(lst, patch):
    """ A patch is [(old_val, increment),...]. No checks made.
    If an old_val is not in lst, returns None.
    >>> print(patch_list([1, 2, 3], [(2, 42), (3, 43)]))
    [1, 44, 46]
    >>> print(patch_list([1, 2, 3], [(42, 2)]))
    None
    """
    try:
        new_lst = lst[:]
        for ov, nv in patch:
            idx = new_lst.index(ov)
            new_lst[idx] += nv
        return new_lst
    except ValueError:  # raised by lst.index(ov) if ov not in lst
        return None



################### Patchers #####################################
class DummyPatcher(object):
    """Do nothing. Simply exposes the functionality for the callers."""
    def __init__(self):
        self.name = "blank"

    def generate_new(self, fs_pairs):
        return fs_pairs

    def context_factor(self, fs_pairs, new_pairs):
        return 1.



class CreAnn(DummyPatcher):
    """ Create/annihilate a pair close to a Fermi level. """
    def __init__(self, N, low_en_range):
        self.name = "create/annihilate_pair"
        self.N, self.low_en_range = N, low_en_range

        if FL(N)[0]+low_en_range >= FL(N)[1]-low_en_range:
            raise RuntimeError("cre/ann: ranges overlap : range=%s\n " 
                    % low_en_range)


    def generate_new(self, fs_pairs):
        Q = np.random.randint(self.low_en_range)

        # L- or R- Fermi point
        if np.random.rand()>0.5: 
            fl = FL(self.N)[1]
            s = 1
        else:
            fl = FL(self.N)[0]
            s = -1

        # create or annihilate
        if np.random.rand() > 0.5:
            new_pairs = fsPairs( fs_pairs.h + [fl-s*Q], 
                                 fs_pairs.p + [fl+s*(1+Q)] )
            return new_pairs

        else:
            new_pairs = fsPairs( fs_pairs.h[:], fs_pairs.p[:] )
            try:
                new_pairs.h.remove(fl-s*Q) 
                new_pairs.p.remove(fl+s*(1+Q)) 
                return new_pairs
            except ValueError:
                return None



class MoveFSHole(DummyPatcher):
    """Move a hole in the Fermi sea. """
    def __init__(self, move_range):
        self.name = "move_fs_hole"
        self.planck = planck(1./move_range)

    def generate_new(self, fs_pairs):
        if not fs_pairs.h:
            return None

        delta = self.planck.rvs()+1
        sign = 1 if np.random.rand() > 0.5 else -1

        h = randomchoice(fs_pairs.h)
        patch = [(h, delta*sign)]

        new_pairs = fsPairs( patch_list( fs_pairs.h, patch ), 
                                fs_pairs.p[:] )
        return new_pairs



class MoveFSParticle(DummyPatcher):
    """Move a particle above the Fermi sea. """
    def __init__(self, move_range):
        self.name = "move_fs_particle"
        self.planck = planck(1./move_range)

    def generate_new(self, fs_pairs):
        if not fs_pairs.p:
            return None

        delta = self.planck.rvs()+1
        sign = 1 if np.random.rand() > 0.5 else -1

        p = randomchoice(fs_pairs.p)
        patch = [(p, delta*sign)]

        new_pairs = fsPairs(fs_pairs.h[:],
                               patch_list( fs_pairs.p, patch ))
        return new_pairs



class UmklappHole(DummyPatcher):
    """Umklapp a hole in the Fermi sea."""
    def __init__(self):
        self.name = "umklapp_hole"

    def generate_new(self, fs_pairs):
        if not fs_pairs.h:
            return None

        h = randomchoice(fs_pairs.h)
        patch = [(h, -2*h-1)]
        new_pairs = fsPairs( patch_list( fs_pairs.h, patch ),
                                fs_pairs.p[:] )
        return new_pairs



class ScatterHP(DummyPatcher):
    """Move a hole & a particle. """
    def __init__(self, move_range):
        self.name = "scatter_hp"
        self.planck = planck(1./move_range)

    def generate_new(self, fs_pairs):
        if not fs_pairs.h:
            return None

        delta = self.planck.rvs()+1
        sign = 1 if np.random.rand() > 0.5 else -1

        h, p = randomchoice(fs_pairs.h), randomchoice(fs_pairs.p)
        patch_h, patch_p = [(h, delta*sign)], [(p, delta*sign)]

        new_pairs = fsPairs( patch_list( fs_pairs.h, patch_h),
                                patch_list( fs_pairs.p, patch_p) )
        return new_pairs



class ScatterPP(DummyPatcher):
    """Move two particles: [-100, 100] <-> [-110, 110] """
    def __init__(self, move_range):
        self.name = "scatter_pp"
        self.planck = planck(1./move_range)

    def generate_new(self, fs_pairs):
        if len(fs_pairs.p)<2:
            return None

        delta = self.planck.rvs()+1
        sign = 1 #if np.random.rand() > 0.5 else -1

        p1, p2 = randomchoice(fs_pairs.p), randomchoice(fs_pairs.p)
        while p1 == p2:
            p2 = randomchoice(fs_pairs.p)

        patch = [(p1, delta*sign), (p2, -delta*sign)]  

        new_pairs = fsPairs(fs_pairs.h[:], 
                            patch_list(fs_pairs.p, patch))
        return new_pairs


class ScatterHH(DummyPatcher):
    """Scatter two holes: [-5, 3] <-> [-3, 1] """
    def __init__(self, move_range):
        self.name = "scatter_hh"
        self.planck = planck(1./move_range)

    def generate_new(self, fs_pairs):
        if len(fs_pairs.p)<2:
            return None

        delta = self.planck.rvs()+1

        h1, h2 = randomchoice(fs_pairs.h), randomchoice(fs_pairs.h)
        while h1 == h2:
            h2 = randomchoice(fs_pairs.h)

        patch = [(h1, delta), (h2, -delta)]  
        new_pairs = fsPairs( patch_list(fs_pairs.h, patch), 
                             fs_pairs.p[:])
        return new_pairs



class UmklScatterHP(DummyPatcher):
    """Umklapp a hole & move a particle. """
    def __init__(self):
        self.name = "umkl_scatter"

    def generate_new(self, fs_pairs):
        if not fs_pairs.h:
            return None

        h, p = randomchoice(fs_pairs.h), randomchoice(fs_pairs.p)
        Q = -2*h-1
        patch_h, patch_p = [(h, Q)], [(p, Q)]

        new_pairs = fsPairs( patch_list( fs_pairs.h, patch_h ),
                             patch_list( fs_pairs.p, patch_p ) )
        return new_pairs


