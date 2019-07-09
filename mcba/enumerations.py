from __future__ import division, print_function, absolute_import
from itertools import combinations, chain, count

import numpy as np

from mcba.abstract_walker import BasicWalker
from mcba.models.impurity.partitions import (fsPairs, pre_hash, fermi_sea, 
        a_partition, sum_deltas )
from mcba.db import row_iterator
from mcba.helpers import roundrobin 


#############################################################################
##################  Straightforward enumeration #############################
#############################################################################
#
# @DEPRECATE; is this buckets-based. Is used @ test_mc_enum, where else? 
#
from mcba.models.impurity import Par, SingleImpurity
from mcba.models.partitions_base import pairs_from_buckets

def is_in_range(buckets, par):
    """
    >>> from mcba.models.impurity import Par
    >>> par = Par(N=3)
    >>> is_in_range(np.array([-2, -1, 0, 1]), par), is_in_range(np.array([-1, 0, 1, 2]), par)
    (True, False)
    """
    return par.m_q - par.N-1 <= np.sum(buckets) <= par.m_q


def generate_partitions(par, nmin, nmax):
    """Generate all compatible partitions with nmin < n_t < nmax:
    >>> from mcba.models.impurity import Par
    >>> par = Par(N=3)
    >>> for p in generate_partitions(par,-2,2): print(p)
    [-2 -1  0  1]
    >>> for p in generate_partitions(par, -2,3): print(p)
    [-2 -1  0  1]
    [-2 -1  0  2]
    [-2 -1  1  2]
    [-2  0  1  2]
    """
    return (np.array(x) for x in combinations(range(nmin, nmax), par.N +1)
             if is_in_range(x, par)
           )

def direct_enumerate(par, nmin, nmax, verbose = True):
    """
    #>>> from mcba.models.impurity import Par, SingleImpurity
    #>>> from mcba.models.impurity.partitions import is_singular
    #>>> par = Par(N=3,L=29,V=1,m_q=2)
    #>>> model = SingleImpurity(par)
    #>>> sols = direct_enumerate(par, -2, 3)
    #.....
    #>>> for s in sols: print("%s %.3f %.3f"% (s, sols[s]["FSfq"], sols[s]["c"]))
    (1, 2) -0.738 1.392
    (-2, 2) 0.500 -inf
    () -0.500 inf
    (0, 2) 0.189 0.217
    (-1, 2) -0.116 -0.959

    #>>> print("sum rule: %.4f" % model.sum_overlaps(sols.itervalues()))
    sum rule: 0.8436
    """
    model = SingleImpurity(par)

    sols = dict()
    for buckets in generate_partitions(par, nmin, nmax):
        if verbose: 
            print(".", end='')
        fs_pairs = pairs_from_buckets(buckets)
        sol = model.calculate(fs_pairs)
        sols[pre_hash(fs_pairs)] = sol
    if verbose: 
        print()
    return sols



###############################################################################
##################### FS-pairs based enumeration ##############################
###############################################################################

def thread_holes(particles, model):
    """Given a set of particles, generate partitions for all
        possible *valid* hole combinations."""
    fs = fermi_sea(model.par.N)
    for p in particles:
        assert p not in fs

    num = len(particles)
    gen_holes = combinations(fs, num)
    for holes in gen_holes:
        fs_pairs = fsPairs(h=list(holes), p=particles)
        if model.is_valid(fs_pairs):
            yield fs_pairs



"""
Successive partitions for num_p>=2 are generated like this:
* enumerate the integers n outside the FS by counting outwards.
* define the `support` given a `limit` x as a set of enumerated integers prior to x.
Now, put a particle at a position x. Then list all combinations of num_p-1 particles 
over the support given x.

Here's a sketch ("-" is the Fermi sea): 
  oo--------------ox
 xoo--------------oo
 ooo--------------oox
 etc 
 where at each step we place a particle at "x" and sum over 
 num_p-1 particles positioned at "o"-s
"""


def count_outwards(N, num=None):
    """Count outwards from the FS, generate exactly `num` values.
    
    >>> for x in count_outwards(3, 2): print(x, '', end='')
    2 -3 
    >>> for x in count_outwards(3, 3): print(x, '', end='')
    2 -3 3 
    >>> for x in count_outwards(4, 2): print(x, '', end='')
    -3 3 
    >>> for x in count_outwards(4, 3): print(x, '', end='')
    -3 3 -4 
    >>> for x in count_outwards(4):
    ...    if x>42: raise StopIteration
    Traceback (most recent call last):
        ...
    StopIteration
    """
    fs = fermi_sea(N)
    counter = chain.from_iterable(zip(count(), (-z-1 for z in count())))
    x, n = 0, 0
    while num is None or n < num:
        x = next(counter)
        if x not in fs:
            n += 1
            yield x



def gen_support(N, num_start, num_stop=None):
    """Generate the `support` by counting outwards.

    >>> for x, supp in gen_support(3, 0, 4): print ((x, supp), end=' ') 
    (2, []) (-3, [2]) (3, [2, -3]) (-4, [2, -3, 3]) 
    >>> for x, supp in gen_support(3, 1, 4): print ((x, supp), end=' ')
    (-3, [2]) (3, [2, -3]) (-4, [2, -3, 3]) 

    >>> for x, supp in gen_support(4, 0, 4): print ((x, supp), end=' ')
    (-3, []) (3, [-3]) (-4, [-3, 3]) (4, [-3, 3, -4]) 
    >>> for x, supp in gen_support(4, 2, 4): print ((x, supp), end=' ')
    (-4, [-3, 3]) (4, [-3, 3, -4]) 

    NB: Badargs are treated quietly:
    >>> for x, supp in gen_support(3, 0, 0): print (x, supp, end='')
    >>> for x, supp in gen_support(4, 0, -2): print (x, supp, end='')
    >>> for x, supp in gen_support(4, 8, 1): print (x, supp, end='')
    """
    support = []
    for j, x in enumerate(count_outwards(N, num_stop)):
        if j >= num_start: 
            yield x, support
        support.append(x)


def gen_particles(num_p, tpl): 
    x, support = tpl
    if num_p < 2:
        raise RuntimeError("gen_particles w/ num_p=%s"%num_p)
    else:
        for y in combinations(support, num_p-1):
            yield list((x,) + y)


def gen_partitions_1(model):
    """Generate partition for num_p=1."""
    N = model.par.N
    minn = model.par.m_q - N -1
    maxx = model.par.m_q + N
    for p in range(minn, maxx + 1):
        if p not in fermi_sea(N):
            for fs_pairs in thread_holes([p], model):
                yield p, fs_pairs



def gen_partitions(num_p, model, num_start, num_stop):
    """
    Generate partitions for num_p>1 pairs for successive support sizes.
    """
    assert num_p >= 0
    if num_p == 0:
        yield 0, fsPairs(h=[], p=[])
    elif num_p == 1:
        for x, fs_pairs in gen_partitions_1(model):
            yield x, fs_pairs
    else:
        for x, supp in gen_support(model.par.N, num_start, num_stop):
            for p in gen_particles(num_p, (x, supp)):
                for fs_pairs in thread_holes(p, model):
                    yield x, fs_pairs



def gen_oneparametric(model, fs_pairs0):
    """Generate partitions for a one-parametric family of fs_pairs, where
    * holes are fixed
    * all but one particle is fixed
    * the position of the remaining particle is a free parameter.

    >>> from mcba.models.impurity import Par, SingleImpurity
    >>> par = Par(N=3, m_q=1)
    >>> model = SingleImpurity(par)
    >>> for p in gen_oneparametric(model, fsPairs(h=[1],p=[])):
    ...    print(p) 
    (2, fsPairs(h=[1], p=[2]))
    (3, fsPairs(h=[1], p=[3]))
    (4, fsPairs(h=[1], p=[4]))
    >>> for p in gen_oneparametric(model, fsPairs(h=[0], p=[])):
    ...    print(p)
    (2, fsPairs(h=[0], p=[2]))
    (3, fsPairs(h=[0], p=[3]))
    """
    summ0 = -sum_deltas(fs_pairs0, model.par)
    maxx = summ0 
    minn = maxx - model.par.N-1
    for p in range(minn, maxx+1):
        fs_pairs = fsPairs(h=fs_pairs0.h, p=fs_pairs0.p+[p])
        if model.is_valid(fs_pairs):
            yield p, fs_pairs


##########################################################
class EnumWalker(BasicWalker):
    """
    Generates partitions by a direct enumeration (see below for details).
   Basic usage:
    >>> from mcba.models.impurity import Par, SingleImpurity
    >>> par = Par(N=3)
    >>> model = SingleImpurity(par)
    >>> from mcba.enumerations import EnumWalker
    >>> walker = EnumWalker(model)
    >>> walker.walk()
    ... # doctest: +SKIP

	This will quit the interactive interpreter once the walker is done: 
    this walker hasn't really been designed for interactive work.
    
    The full constructor signature is:
        EnumWalker(par, **kwargs)    

    Keyword arguments and their defaults are:
           max_num_p      :  3 (enumerate the states with the 0, 1, ..., max_num_p
                                particle-hole pairs)
           max_limit      :  None (unless is None, the simulation stops if this is reached 
                             irrespective of num_sweeps or threshold, see below)
           threshold      :  0.995 (this is the target for the sum rule:
                             the simulation stops when either sweeps == num_sweeps,
                             or  \sum |<FS|fq>|^2 > threshold,
                             whichever comes first.)

    Additionally, these kwargs are inherited from AbstractWalker via BasicWalker:
           num_sweeps      : numpy.inf
           steps_per_sweep : 100
           therm_sweeps    : 0
           checkp_sweeps   : 42
           printout_sweeps : numpy.inf
           seed            : None (if None, random number generator
                             relies on the numpy convention: it takes the seed
                             from /dev/random or clock)
           keep_rcache     : False
           lower_cutoff    : 0 (only store cnfs with weight>lower_cutoff)
           db_fname        : ":memory:" (default is an in-memory sqlite DB)
           db_prefix       : "en" (used for the DB id string)
           store_roots     : True (if False, roots will not be saved to the DB)

    If keep_rcache=False, rcache is reset at checkpoints. Setting to True
    might not be very memory-friendly. 

    Successive partitions for num_p>=2 are generated like this:
    * enumerate the integers outside the FS by counting outwards.
    * define the 'support' given a 'limit' x as a set of enumerated
      integers prior to x.
    Now, put a particle at a position x. Then list all combinations of num_p-1 
    particles over the support given x.

    Here's a sketch ("-" is the Fermi sea): 
      oo--------------ox
     xoo--------------oo
     ooo--------------oox
     etc 
     where at each step we place a particle at "x" and sum over 
     num_p-1 particles positioned at "o"-s
    """
    def __init__(self, model, max_num_p=3, max_limit=None, db_prefix="en", **kwargs):
        super(EnumWalker, self).__init__(model, db_prefix=db_prefix, **kwargs)
        self.name = "EnumWalker"

        self.max_num_p = max_num_p
        self.max_limit = max_limit
        self.db_prefix = db_prefix

        self.fs_pairs = a_partition(self.model.par)
        
        # **** Known bugs *****
        # Generators for 0 & 1 fs_pairs are a bit special: 
        # 1. they do not honor the curr/max_limit. Hence, if restarted, 
        #    they redo the work from scratch. Which is annoying, but isn't 
        #    too big of a deal (they are finite, after all). Hence, WONTFIX
        # 2. Is there a way of telling if gen_partitions_01(1) is exhausted.
        #    is_work_done() only looks at curr_limit, hence it'll report 
        #    "working" even when it's actually finished. Annoying, yes. 
        #    Still, WONTFIX 

        ################## Set up the DB and where to start from ############
        if not self.is_fresh:
            xxx = 0
            for row in row_iterator(self.db_handle):
                particles = row["partition"].p
                if len(particles)>1:
                    xxx = max(xxx, max(particles, key = lambda x: x**2),
                                key = lambda x: x**2)
            self.curr_limit = xxx
        else:
            self.curr_limit = 0

        #################### set up the generators #########################
        gens = [gen_partitions(num, self.model, self.curr_limit, self.max_limit)
                for num in range(0, self.max_num_p+1) ] # including max_num_p
        self.generator = roundrobin(*gens)

        # do we actually need to do anything?
        if self.is_work_done():
            print("nothing to be done: ", self.is_work_done())
            self.finalize()


    def do_step(self):
        """Round-robin over num_p-s."""
        self.curr_limit, self.fs_pairs = next(self.generator)
        
        # fs_pairs can still be invalid: e.g. q>k_F fsPairs([],[])
        if self.model.is_valid(self.fs_pairs):
            cnf = self.model.calculate(self.fs_pairs)
            if cnf["FSfq"]**2 >= self.lower_cutoff:
                self.rcache[pre_hash(self.fs_pairs)] = cnf



    def is_work_done(self): 
        # self.max_limit can be None, hence need to guard against it
        return self.sum_overlaps() > self.threshold or\
               self.curr_limit >= (self.max_limit if self.max_limit else np.inf)


    def printout(self):
        print("\n************** sweeps = ", self.sweeps)
        print(self.model.long_print_par())
        print("current limit = ", self.curr_limit)
        print("total cnf count = ", self.num_cnf(), 
                "rcache size = ", len(self.rcache))
        print("\\sum |<FS|fq>|^2 = {0:.4}".format(1.*self.sum_overlaps()))
        print("\\sum <fq| P_up |fq> = {0:.4}".format(1.*self.sum_P()))



#######################################
if __name__ == "__main__":
    import doctest
    doctest.testmod()

