from __future__ import division, print_function, absolute_import

import numpy as np

from .abstract_walker import BasicWalker
from .abstract_update import Update
from .models.partitions_base import fsPairs, pre_hash, from_prehash
from .db import row_iterator

from .updates import (DummyPatcher, CreAnn, MoveFSHole, MoveFSParticle,
        UmklappHole, UmklScatterHP, ScatterHP, ScatterPP, ScatterHH )


class Walker(BasicWalker):
    """
    Generates partitions in a Metropolis MC random walk. 

    Basic usage:    
    >>> from mcba.models.impurity import Par, SingleImpurity
    >>> par = Par(N=3)
    >>> model = SingleImpurity(par)
    >>> from mcba.walker import Walker
    >>> walker = Walker(model)
    >>> walker.walk()
    ... #doctest: +SKIP

    This will quit the interactive interpreter once the walker's done: walker 
    hasn't really been designed for interactive work.

    The full constructor signature is:
        Walker(model, fs_pairs=None, **kwargs)

    Here (optional) fs_pairs is the starting partition.

    Keyword arguments and their defaults are:
        Basic MC process control (inherited from AbstractWalker via BasicWalker)
           num_sweeps      : numpy.inf
           steps_per_sweep : 100
           therm_sweeps    : 0
           checkp_sweeps   : 42
           printout_sweeps : 42 (use None if not desired) 
           seed            : None (if None, random number generator 
                             relies on the numpy convention: it takes the seed 
                             from /dev/random or clock)
           keep_rcache     : False
           threshold       : 0.995 (this is the target for the sum rule:
                             the simulation stops when either sweeps==num_sweeps,
                             or  \sum |<FS|fq>|^2 > threshold,
                             whichever comes first.)
           lower_cutoff    : 0 (only store cnfs with weight > lower_cutoff)
           db_fname        : ":memory:" (default is an in-memory sqlite DB)
           db_prefix       : "mc" (used for the DB id string)
           store_roots     : True (if False, roots will not be saved to the DB)

        MC updates furthermore, use these ones:
           low_en_range    : 5 if par.N>11 else 1 (cf CreAnn pair update) 
           jump_length     : 5 (cf Move- & Scatter-type updates)
           reset_freq      : 1 restart the walker from a random cache entry once
                               every this many sweeps

    Holds two caches:
    * lookup_cache is {fs_pairs : FSfq}, and
    * rcache, which is { fs_pairs : {"FSfq", "c", "P", "roots"} }

    If keep_rcache=False, rcache is reset at checkpoints. Setting to True
    might not be very memory-friendly.
    """
    def __init__(self, model, fs_pairs=None, jump_length=5, low_en_range=None, 
                       reset_freq=1, **kwargs):
        super(Walker, self).__init__(model, **kwargs)
        self.name = "MC Walker"

        self.jump_length = jump_length
        if low_en_range is None:
            self.low_en_range  = 5 if model.par.N > 11 else 1
        else:
            self.low_en_range = low_en_range
            
        if reset_freq is not None:
            self.periodic_actions.append({"action": self.reset,
                                          "freq": reset_freq})

        ################## Get the lookup_cache ######################
        self.lookup_cache = {}
        if not self.is_fresh:
            print("\nloading the cache...", end='')
            for row in row_iterator(self.db_handle):
                prehash = pre_hash(row["partition"])
                self.lookup_cache[prehash] = row["FSfq"]
            print("done. Start w/ ", self.num_cnf()," entries.")

        ############ set up the starting configuration ##################
        self.fs_pairs = fs_pairs

        if self.fs_pairs is None:
            if self.lookup_cache:
                self.reset()
            else:
                self.fs_pairs = self.model.a_partition()
        else:
            # being paranoid. Not asserts since I want these 
            # checks to work w/ python -O too.
            if not isinstance(fs_pairs, fsPairs) or \
                    not self.model.is_valid(fs_pairs):
                raise ValueError("Cannot start MC process from fs_pairs= " + 
                        str(fs_pairs))
        assert self.model.is_valid(self.fs_pairs)

        cnf = self.model.BA_solver.solve(self.fs_pairs)
        if cnf["c"] == np.inf or cnf["c"] == -np.inf:
            raise ValueError("Cannot start MC process from a singular configuration. Stop.")

        dummy_upd = Update(self.model, DummyPatcher())
        self.fs_pairs, cnf = dummy_upd(self.fs_pairs, self.lookup_cache)

        key = pre_hash(self.fs_pairs)
        if cnf and key not in self.lookup_cache:
            self.lookup_cache[key] = cnf["FSfq"]
            self.rcache[key] = dict( (k, cnf[k]) for k in ["FSfq", "P", "roots", "c"] )

        ################### set up the updates ##############################
        self.updates = [
                Update(self.model, CreAnn(self.model.par.N, self.low_en_range)),
                Update(self.model, MoveFSHole(self.jump_length)),
                Update(self.model, MoveFSParticle(self.jump_length)),
                Update(self.model, UmklappHole()),
                Update(self.model, UmklScatterHP()),
                Update(self.model, ScatterHP(self.jump_length)),
                Update(self.model, ScatterPP(self.jump_length)), 
                Update(self.model, ScatterHH(self.jump_length)),
                       ]

        # do we actually need to do anything?
        if self.is_work_done():
            print("nothing to be done for this one, exiting.")
            self.finalize()



    def do_step(self):
        """ Dispatch the main MC actions."""
        for update in self.updates:
            self.fs_pairs, cnf = update(self.fs_pairs, self.lookup_cache)

            key = pre_hash(self.fs_pairs)
            if cnf and key not in self.lookup_cache:
                # first time here, cache it
                self.lookup_cache[key] = cnf["FSfq"]
                self.rcache[key] = dict( (k, cnf[k]) 
                        for k in ["FSfq", "P", "roots", "c"] )

    def is_work_done(self):
        return self.sweeps >= self.num_sweeps or \
               self.sum_overlaps() > self.threshold



    def reset(self):
        """Restart the walker from a random cnf from the cache."""
        n = np.random.randint(self.num_cnf())
        bk = self.lookup_cache.keys()
        for _ in range(n):
            next(bk)
#        self.fs_pairs = from_prehash(next(bk))     # previous
        self.fs_pairs = from_prehash(tuple(bk)) 



    def printout(self):
        print("\n***************** sweeps = ", self.sweeps, "\t(", 
                self.steps_per_sweep, " steps each)")
        print(self.model.long_print_par())
        print("total cnf count = ", self.num_cnf(), "rcache size = ", 
                len(self.rcache))
        print("\\sum |<FS|fq>|^2 = {0:.4g}".format(self.sum_overlaps()))
        print("\\sum  <fq | P_up | fq> = {0:.4g}".format(self.sum_P()))
        print("self.fs_pairs = ", self.fs_pairs)
        print("addr/acpt: ")
        for upd in self.updates:
            print("\t", upd.counter)
        print('\n')



    def num_cnf(self):
        return len(self.lookup_cache)


