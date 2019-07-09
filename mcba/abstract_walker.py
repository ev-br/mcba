from __future__ import division, print_function, absolute_import

import numpy as np

import mcba.db as db
from .models.impurity.partitions import pre_hash



class AbstractWalker(object):
    def __init__(self, num_sweeps=np.inf, therm_sweeps=0,
                       printout_sweeps=42, checkp_sweeps=42, 
                       seed=None, **kwargs):
        """
        Bare-bones MC walker. To be inherited from.

        Valid kwargs are:
           num_sweeps      : numpy.inf
           therm_sweeps    : 0
           checkp_sweeps   : 42
           printout_sweeps : 42 
           seed            : None (if None, random number generator 
	                         relies on the numpy convention: it takes the seed 
                             from /dev/random or clock)
        
        Usage: Implement do_step(). Define steps_per_sweep.
        Override printout(), checkpoint(), and, if desired,
        is_thermalized(), finalize(), is_work_done().
        
        For having some action done once every so many sweeps, append 
        to periodic_actions (which is a list), a dict of 
        {"action", "freq"} where action is a callable and freq is in sweeps. 

        Use gen_walk() for sweep-by-sweep run, and walk() otherwise.
        """
        try:
            super(AbstractWalker, self).__init__(**kwargs)
        except TypeError:
            print("\n *** Unrecognized keyword arguments: ", kwargs, " *** \n")
            raise

        self.num_sweeps = num_sweeps
        self.therm_sweeps = therm_sweeps
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        self.sweeps, self.step = 0, 0
        
        self.periodic_actions = []
        if checkp_sweeps is not None:
            self.periodic_actions.append({"action": self.checkpoint, 
                                          "freq": checkp_sweeps})
        if printout_sweeps is not None:
            self.periodic_actions.append({"action": self.printout,
                                          "freq": printout_sweeps})


    def is_thermalized(self):
        return self.sweeps >= self.therm_sweeps

    #def do_sweep(self): pass

    def checkpoint(self): pass

    def printout(self): pass

    def finalize(self):
        self.checkpoint()

    def is_work_done(self):
        return self.sweeps >= self.num_sweeps

    def status(self):
        if self.is_work_done():
            return "done"
        else:
            return "running"


    def walk(self):
        """User-callable: walk the walk."""
        for _ in self.gen_walk():
            pass
        self.finalize()


    def gen_walk(self):
        """Step-by-step evolution of the walk."""
        while not self.is_work_done():
            self.sweeps += 1
            for self.step in range(self.steps_per_sweep):
                self.do_step()
                yield self

            # prntout, checkpoint, reset, expensive_measurements etc
            for action in self.periodic_actions:
                if self.sweeps % action["freq"] == 0:
                    action["action"]() 



class BasicWalker(AbstractWalker):
    """Instantiates solvers, receives par, sets up the DB."""
    def __init__(self, model, keep_rcache=False, lower_cutoff=0, threshold=0.995,
                       db_fname=":memory:", db_prefix="mc", verbose_logg=True,
                       store_roots=True, steps_per_sweep=None, **kwargs):
        super(BasicWalker, self).__init__(**kwargs)

        self.model = model
        self.db_fname, self.db_prefix = db_fname, db_prefix
        self.store_roots, self.verbose_logg = store_roots, verbose_logg
        self.keep_rcache = keep_rcache
        self.threshold, self.lower_cutoff = threshold, lower_cutoff

        if steps_per_sweep is None:
            self.steps_per_sweep = self.model.par.N +1
        else:
            self.steps_per_sweep = steps_per_sweep

        ################## Set up the DB & get the caches ######################
        self.db_handle, self.is_fresh = db.setup_db(self.model, self.db_fname, 
                self.seed, prefix=self.db_prefix)

        self.rcache = {}
        if not self.is_fresh:
            # being paranoid: make sure the DB is OK first:
            db.verify_DB(self.db_handle, self.model,
                     num=10, check_roots=self.store_roots, atol=1e-8, rtol=1e-8)

            self.rcache = {}
            if self.keep_rcache:
                print("\nloadign the full_cache...", end='')
                names = ["FSfq", "c", "P", "roots"]
                for row in db.row_iterator(self.db_handle):
                    prehash = pre_hash(row["partition"])
                    entry = dict( (name, row[name]) for name in names)
                    self.rcache[prehash] = entry
                print("done")



    def sum_overlaps(self):
        if self.keep_rcache:
            sumr = self.model.sum_overlaps(self.rcache.values())
        else:
            # get it from disk
            sumr = self.model.sum_overlaps(db.row_iterator(self.db_handle))
        return sumr



    def sum_P(self):
        if self.keep_rcache:
            P = self.model.av_momt(self.rcache.values())
        else:
            # get it from disk
            P = self.model.av_momt(db.row_iterator(self.db_handle))
        return P



    def num_cnf(self):
        if self.keep_rcache:
            return len(self.rcache)
        else:
            return db.get_DBsize(self.db_handle)



    def checkpoint(self):
        print("\n *** checkpointing, do NOT terminate...", end='')
        db.dump_cache(self.db_handle, self.rcache, self.store_roots)
        num_cnf = self.num_cnf()
        sum_overl = self.sum_overlaps()
        sum_p = self.sum_P()
        db.update_results(self.db_handle, sum_overl, sum_p, self.status())
        
        mesg = "sum<FSfq>2= {0}  P= {1}  cachesize= {2}".format(sum_overl, 
                    sum_p, num_cnf)
        db._log_trans(self.db_handle, mesg)

        if not self.keep_rcache:  
            self.rcache = {}  
        print("done")



    def finalize(self):
        self.checkpoint()
        print("\nFINALLY,\n  total distinct confs: ", self.num_cnf())
        print("\\sum |<FS|fq>|^2 = ",  self.sum_overlaps())
        print("\\sum  <fq | P_up | fq> = ", self.sum_P())
        print("FINALIZE:", self.status(),"\n")


