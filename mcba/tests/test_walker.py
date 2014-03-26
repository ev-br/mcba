import numpy
import os
from nose.tools import raises #, nottest

import mcba
from mcba.walker import Walker
import mcba.db as db
import mcba.models.partitions_base as pt
import mcba.helpers as hp
from mcba.models.impurity import Par, SingleImpurity

def remove_file(fname):
    """A helper for setup/teardown.""" 
    if os.path.exists(fname):
        os.remove(fname)



class TestWalkerCtor(object):
    attrib = "quick"

    def setup(self):
        par = Par(N=3,m_q=1, V=numpy.inf)
        self.model = SingleImpurity(par)
        self.mc_dict = {
            "num_sweeps"      : 10,
            "steps_per_sweep" : 1000,
            "therm_sweeps"    : 10,
            "therm_sweeps"   : 10,
            "printout_sweeps" : None,
            "db_fname"        : None,
            "low_en_range"    : 1,
            "seed"            : 42
                       }

    @raises(ValueError)  # wrong type of pairs
    def test_buckets(self):
        walker = Walker(self.model, fs_pairs=[-1, 0, 1], **self.mc_dict)


    @raises(ValueError)   # can't have num_pairs > num_buckets
    def test_cant_be_pairs(self):
        walker = Walker(self.model, fs_pairs= ([0]*5, [0]*5), **self.mc_dict)


    @raises(ValueError)
    def test_outta_range(self):
        walker = Walker(self.model, fs_pairs=pt.fsPairs([-2], [42]), **self.mc_dict)

    @raises(ValueError)
    def test_singular(self):
        walker = Walker(self.model, fs_pairs=pt.fsPairs([-2], [4]), **self.mc_dict)

    @raises(TypeError)
    def test_kwargs(self):
        walker = Walker(self.model, foo='bar', **self.mc_dict)

    @raises(TypeError)
    def test_kwargs2(self):
        dct = hp.copydict(self.mc_dict, {"num sweeps": 42})  #space, not underscore
        walker = Walker(self.model, **dct)




class TestWalkerWalk(object):
    attrib = "quick"

    def setup(self):
        par = Par(N=5, m_q=1, V=numpy.inf)
        self.model = SingleImpurity(par)
        self.mc_dict = {
            "num_sweeps"      : 1,
            "steps_per_sweep" : 1,
            "therm_sweeps"    : 0,
            "checkp_sweeps"   : 5,
            "printout_sweeps" : 5,
            "db_fname"        : "testdb.db",
            "low_en_range"    : 1,
            "seed"            : 42
                       }
        remove_file(self.mc_dict["db_fname"])

    def teardown(self):
        remove_file(self.mc_dict["db_fname"])

    def test_walk(self):
        walker = Walker(self.model, **self.mc_dict)
        walker.walk()
        assert os.path.exists(self.mc_dict["db_fname"])



class TestWalkerWalkTheshold(object):
    attrib = "quick"

    def setup(self):
        par = Par(N=5, m_q=1, V=numpy.inf)
        self.model = SingleImpurity(par)
        self.mc_dict = {
            "num_sweeps"      : numpy.Inf,
            "steps_per_sweep" : 5,
            "checkp_sweeps"   : 10,
            "printout_sweeps" : None,
            "db_fname"        : None,
            "low_en_range"    : 1,
            "seed"            : 42, 
            "threshold"       : 0.7,
                       }

    def test_walk(self):
        """Should get terminated via the threshold"""
        walker = Walker(self.model, **self.mc_dict)
        walker.walk()

        c = walker.db_handle.execute("SELECT * from mcrun_data")
        sum_overl = self.model.sum_overlaps(c)
        assert sum_overl > 0.7



class TestWalkerRestart(object):
    attrib = "quick"

    def setup(self):
        par = Par(N=5,m_q=1, V=numpy.inf)
        self.model = SingleImpurity(par)
        self.mc_dict = {
            "num_sweeps"      : 1,
            "steps_per_sweep" : 10,
            "therm_sweeps"    : 0,
            "checkp_sweeps"   : 100,
            "printout_sweeps" : 5,
            "db_fname"        : "test.db",
            "low_en_range"    : 1,
            "seed"            : 42,
            "keep_rcache"     : True,
                       }
        remove_file(self.mc_dict["db_fname"])

    def teardown(self):
        print "teardown"
        remove_file(self.mc_dict["db_fname"])


    def test_restart(self):
        walker = Walker(self.model, **self.mc_dict)
        walker.walk()
        sum_overl0 = self.model.sum_overlaps(walker.rcache.itervalues())

        mc_dict = hp.copydict(self.mc_dict, {"threshold": 0.9, "num_sweeps": 5})
        walker2 = Walker(self.model, **mc_dict)
        walker2.walk()
        sum_overl1 = self.model.sum_overlaps(walker.rcache.itervalues())
        assert sum_overl1 >= sum_overl0

        #FIXME: keep_rcache=True & restarts should be a separate one

        # Make sure the DB only has unique cnfs
        wpart = [pt.pre_hash(row["partition"]) for row 
                	in db.row_iterator(walker.db_handle)]
        assert len(wpart) == len(set(wpart))


class TestWalkerEmptyDB(object):
    attrib = "quick"

    def setup(self):
        par = Par(N=3, m_q=1, V=numpy.inf)
        self.model = SingleImpurity(par)
        self.mc_dict = {
            "num_sweeps"      : 1,
            "steps_per_sweep" : 1,
            "therm_sweeps"    : 0,
            "printout_sweeps" : 10,
            "db_fname"        : "test.db",
            "seed"            : 42,
            "low_en_range"    : 1
                       }
        remove_file(self.mc_dict["db_fname"])

    def teardown(self):
        remove_file(self.mc_dict["db_fname"])

    def test_empty_db(self):
        db.setup_db(self.model, self.mc_dict["db_fname"], self.mc_dict["seed"])
        walker = Walker(self.model, **self.mc_dict)
        walker.walk()



class TestWalkerLoadFullCache(object):
    attrib = "quick"

    def setup(self):
        par = Par(N=5, m_q=1, V=numpy.inf)
        self.model = SingleImpurity(par)
        self.mc_dict = {
            "num_sweeps"      : 10,
            "steps_per_sweep" : 10,
            "therm_sweeps"    : 0,
            "checkp_sweeps"   : 10,
            "printout_sweeps" : 10,
            "db_fname"        : "test.db",
            "seed"            : 42,
            "low_en_range"    : 1, 
            "keep_rcache"     : True, 
                       }
        remove_file(self.mc_dict["db_fname"])

    def teardown(self):
        remove_file(self.mc_dict["db_fname"])

    def test_load_full_cache(self):
        walker = Walker(self.model, **self.mc_dict)
        walker.walk()

        sum_rcache = self.model.sum_overlaps(walker.rcache.itervalues())
        sum_db = self.model.sum_overlaps( db.row_iterator(walker.db_handle) )

        numpy.testing.assert_allclose(sum_db, sum_rcache)



class TestWalkerNoRoots(object):
    attrib = "quick"
    def setup(self):
        par = Par(N=3, m_q=1, V=1.)
        self.model = SingleImpurity(par)
        self.mc_dict = {
            "num_sweeps"      : 1500,   # this one is 'quick', see below
            "steps_per_sweep" : 100,
            "therm_sweeps"    : 0,
            "checkp_sweeps"   : 10,
            "printout_sweeps" : 50,
            "db_fname"        : ":memory:",
            "seed"            : 42, 
            "keep_rcache"     : True,
            "threshold"       : 0.92,
            "store_roots"     : False,
        }
        remove_file(self.mc_dict["db_fname"])

    def teardown(self):
        remove_file(self.mc_dict["db_fname"])

    def test_restart(self):
        #run MC
        walker = Walker(self.model, **self.mc_dict)
        walker.walk()

        # keep running
        self.mc_dict["threshold"] = 0.975
        walker2 = Walker(self.model, **self.mc_dict)
        walker2.walk()
        handle = walker2.db_handle

        assert len(walker2.rcache) == \
                handle.execute("SELECT count(*) FROM mcrun_data").fetchone()[0]
        assert all( row["roots"] is None for row in db.row_iterator(handle) )




class TestWalkerSolverAcptDBs(object):
    attrib="acpt"
    """'Acceptance': Recalculate a bunch of cnfs for N=5, 15, 25, 45 and 135."""
    
    fnames = ["baseN5mq2.sqlite", 
              "baseN15mq5.sqlite", 
              "baseN25mq12.sqlite",
              "baseN45mq22.sqlite",
              "baseN135mq67.sqlite",
             ]

    mc_dict = {"db_fname" : None}


    def test_compare_base_DBs(self):

        # where to look for the files
        basepath = os.path.dirname(mcba.__file__)
        basepath = os.path.join(basepath, "tests", "acpt_DBs")
      
        for fname in self.fnames:
            fullpath = os.path.join(basepath, fname)
            assert os.path.exists(fullpath)
            yield self.compare_all, fullpath

    
    def compare_all(self, fname):
        print "reading in: ", fname
        handle, = db.get_handles(fname)
        par = db.get_param(handle)

        model = SingleImpurity(par)
        walker = Walker(model, **self.mc_dict)
        
        # verify all entries
        db.verify_DB(handle, model, num=None, atol=1e-8, rtol=1e-8)     


  

