from __future__ import division, print_function, absolute_import

import numpy as np

from .models.partitions_base import pre_hash


def _ratio(candidate, cnf, key="FSfq"):
    """Metropolis ratio."""
    wca, wcn = candidate[key]**2, cnf[key]**2
    return wca/wcn



class Update(object):
    """An ABC for the updates: holds solvers, mocks the reverse.
    Uses the lookup_cache, but does not update it. 
    
    1. lookup_cache is {fs_pairs : FSfq}
    2. cnf is a dict, the name of the key for caching/ratio is an arg to ctor.
    3. Real work is done in _do_update(), of which a __call__ is an alias.
    4. Patcher is supposed to have at least .generate_new() and .context_factor()
    
    """
    def __init__(self, model, patcher, cnf_key="FSfq"):
        self.model = model
        self.patcher = patcher
        self.counter = AacptCounter(patcher.name)
        self.cnf_key = cnf_key


    def _do_update(self, fs_pairs, lookup_cache):
        self.counter.address()

        new_pairs = self.patcher.generate_new(fs_pairs)

        if new_pairs is None or not self.model.is_valid(new_pairs):
            return fs_pairs, None
        assert self.model.is_valid(new_pairs) 
    
        self.counter.propose()

        cnf = self._lookup(fs_pairs, lookup_cache)
        candidate = self._lookup(new_pairs, lookup_cache)

        ratio = self.patcher.context_factor(fs_pairs, new_pairs)  #FIXME: reverse contex, too
        ratio *= _ratio(candidate, cnf, self.cnf_key)

        if ratio > 1. or np.random.rand() < ratio:
            self.counter.accept()
            return new_pairs, candidate
        else:
            return fs_pairs, None


    # mock_reverse be here
    __call__ = _do_update


    def _lookup(self, fs_pairs, lookup_cache): 
        """Look up fs_pairs in cache. If not found, solve BA eq., return the cnf."""
        # look it up:
        candidate_weight = lookup_cache.get(pre_hash(fs_pairs))

        # if first time here, solve BA eqs, do matrix elements
        if candidate_weight is None:
            candidate = self.model.calculate(fs_pairs) 
        else:
            candidate = {self.cnf_key : candidate_weight}
        return candidate






class AacptCounter(object):
    """Address/accept counter for updates."""
    def __init__(self, name=""):
        self.reset()
        self.name = name
        self.arr = [0, 0, 0]

    def reset(self):
        self.arr = [0, 0, 0]

    def set_name(self, name=""):
        self.name = name

    def address(self):
        self.arr[0] += 1

    def propose(self):
        self.arr[1] += 1

    def accept(self):
        self.arr[2] += 1

    def __str__(self):
        dummy = (self.arr[1]/(self.arr[0]+1e-5), self.arr[2]/(self.arr[0]+1e-5))
        return self.name + ":  {0:.2} / {1:.2}".format( *dummy )


