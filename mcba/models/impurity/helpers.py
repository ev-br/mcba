r"""
Summation helpers: 

* from an iterable of configurations, construct the contributions to 
$\sum_{f_q} | <FS|f_q> |^2$ and $\sum_{f_q} <f|P|f> |<FS|fq>|^2$, 
taking care of the double-counting etc. 

* provide the user-callable summators (sum_overlaps, av_momt)
* to avoid precision loss, use math.fsum
"""
from __future__ import division, print_function, absolute_import

from numpy import inf
from math import fsum
import itertools

from .matrix_elements import energy, det_Cauchy
from .partitions import buckets_from_pairs


def naive_sum_overlaps(sols):
    """To avoid double-counting of the states with c=\pm\infty,
    do NOT include into the sum the states w/ c=+\infty.
    NAIVE summation: no particular order.
    """
    return sum(sols[s]["FSfq"]**2 for s in sols if sols[s]["c"] != inf)


def naive_av_momt(sols):
    """Naive summation: no particular order."""
    return sum(sols[s]["P"] * sols[s]["FSfq"]**2 for s in sols if sols[s]["c"] != inf)



def av_momt(it, dbl_cnt = lambda x: x!=inf):
    """Careful summation of <P>*<FS|f_q>**2, taking care of the double-counting:
    include either c=inf, or c=-inf, but not both.
    """
    return fsum( cnf["P"]*cnf["FSfq"]**2 for cnf in it if dbl_cnt(cnf["c"]) )


def sum_overlaps(it, dbl_cnt=lambda x: x!=inf):
    """Careful summation of <FS|f_q>**2, taking care of the double-counting:
    include either c=inf, or c=-inf, but not both.
    """
    return fsum( cnf["FSfq"]**2 for cnf in it if dbl_cnt(cnf["c"]) )





class cnfProxy(dict):
    """A dict-like proxy object for the roots: 
    * if a key is present, return the corresponding value
    * otherwise, (re)calculate it (model's provided).
    """
    def __init__(self, cnf, model):
        self.update(cnf)
        self.model = model
        
    def __getitem__(self, key):
        try:
            # work around a dumb past decision: roots=None in the DB; TODO
            value = dict.__getitem__(self, key)
            if key == "roots" and value is None:
                raise KeyError
            return value

        except KeyError:
            if key == "partition":
                return self["fs_pairs"]
            if key == "fs_pairs":
                return self["partition"]
            if key == "buckets":
                return buckets_from_pairs(self["fs_pairs"], self.model.par.N)
            if key == "roots":
                rts = self.model.BA_solver.find_roots(self["fs_pairs"], self["c"])
                dict.__setitem__(self, "roots", rts["roots"])  
                return self["roots"]
            if key == "energy":
                ene = energy(self, self.model.par)
                dict.__setitem__(self, "energy", ene)
                return ene
            if key == "detCauchy":
                dC = det_Cauchy(self, basis_state=self.model.basis_state, 
                        par=self.model.par)
                dict.__setitem__(self, "detCauchy", dC)
                return dC
       
        # not a key I know how to deal with, rethrow 
        raise KeyError(key)


