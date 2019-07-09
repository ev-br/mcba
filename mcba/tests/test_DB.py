import os
import sqlite3
import numpy as np
from nose.tools import raises, nottest

import mcba.db as db
from mcba.helpers import copydict
from mcba.models.impurity import Par


""" Unittest the db.
"""

class MockModel(object):
    """setup_db expect a model. Here's mock."""
    def __init__(self, par=None):
        self.par = par if par is not None else Par(N=3)
        self.id_str =  "foobarbaz"
        

def mock_cache():
    """Helper: return a mock cache."""
    cnf = {"roots" : np.array([1., 2., 3., 4.]),  
           "P" : 42., "FSfq" : -42, "c" : np.inf }
    cache = {}
    cache[db.pre_hash(db.fsPairs([],[]))] = cnf
    cache[db.pre_hash(db.fsPairs([0],[42]))] = copydict(cnf, {"FSfq": 0})
    cache[db.pre_hash(db.fsPairs([0],[4]))] = copydict(cnf, {"FSfq": 11})
    return cache


def remove_file(fname):
    """Is useful for setup/teardown."""
    if os.path.exists(fname):
        os.remove(fname)  



class TestHandleUsable(object):
    attrib="quick"

    @raises(AssertionError)
    def test_bare_handle(self):
        handle = db.DBHandleWithID(":memory:", None, None)
        assert handle.is_usable() and handle.is_key_ok()
    
    def test_handle_setup(self):
        mc_dict = {"seed" : 44, "db_fname" : ":memory:", }
        model = MockModel()
        handle, is_fresh = db.setup_db(model, **mc_dict)
        assert handle.is_usable() and handle.is_key_ok()



class TestHandleClosedNotUsable(object):
    attrib = "quick"
    db_fname = "dummy.db"
    
    def setup(self):
        remove_file(self.db_fname)
    def teardown(self):
        remove_file(self.db_fname)

    @raises(AssertionError)
    def test_closed_is_not_usable(self):
        """A handle to a persistent DB is not usable when closed."""
        handle = db.DBHandleWithID(self.db_fname, "foo", 0)
        assert handle.is_usable()

    def test_closed_in_memory(self):
        """A handle to an in-memory DB *is* usable even when closed."""
        handle = db.DBHandleWithID(":memory:", "foo", 0)
        assert handle.is_usable()
        

class TestRoundTrip(object):
    """Make sure a cache roundtrips to/from the DB."""
    attrib = "quick"
    db_fname = ":memory:"

    def test_roundtrip(self):
        model = MockModel()
        cache = mock_cache()

        handle, is_fresh = db.setup_db(model, self.db_fname, seed=42)
        db.dump_cache(handle, cache)

        cols = ["roots", "P", "FSfq", "c"]
        cache_back = {}
        for row in db.row_iterator(handle):
            print("row_ = ", row["partition"])
            cache_back[db.pre_hash(row["partition"])] =\
                dict( (col, row[col]) for col in cols)

        assert sorted(cache_back.keys()) == sorted(cache.keys())


class TestDumpTwice(object):
    attrib = "quick"
    
    db_fname = ":memory:"
    seed = 44

    def test_dump_twice(self):
        fs_pairs = db.fsPairs(h=[], p=[]) 
        cnf = {"roots" : np.array([1., 2., 3., 4.]), 
               "P" : 42., "FSfq" : -42, "c" : np.inf
              }
        rcache = { db.pre_hash(fs_pairs) : cnf }

        # now dump it, twice
        model = MockModel()
        handle, is_fresh = db.setup_db(model, self.db_fname, self.seed)
        db.dump_cache(handle, rcache)
        db.dump_cache(handle, rcache)

        with handle:
            num = handle.execute("""SELECT count(*) FROM mcrun_data;""").fetchone()[0] 
        assert num == 1



class TestCreatePersistent(object):
    attrib = "quick"

    db_fname = "testdb.sqlite"
    seed = 44

    def setup(self):
        self.model = MockModel()
        remove_file(self.db_fname)

    def teardown(self):
        remove_file(self.db_fname)

    def test_cr_anew(self):
        """Create a new DB."""
        db.setup_db(self.model, self.db_fname, self.seed)
        assert os.path.exists(self.db_fname)

    def test_noseed(self):
        """Create a new DB w/ seed = None."""
        db.setup_db(self.model, self.db_fname, self.seed)
        assert os.path.exists(self.db_fname)

    def test_cr_twice(self):
        """Persistent: Double create"""
        db.setup_db(self.model, self.db_fname, self.seed)
        db.setup_db(self.model, self.db_fname, self.seed)
        assert os.path.exists(self.db_fname)

    def test_dbl_context(self):
        """Persistent: Double context"""
        handle, is_fresh =  db.setup_db(self.model, self.db_fname, self.seed)
        with handle:
            with handle:
                handle, is_fresh = db.setup_db(self.model, self.db_fname, self.seed)
            db.get_param(handle)

    def test_log_trans(self):
        """Persistent: Check the commits are done properly."""
        handle, is_fresh = db.setup_db(self.model, self.db_fname, self.seed)
        with handle:
            db._log_trans(handle,"fourty two", None)
        with handle:
            with handle:
                mesg = handle.execute("SELECT * from mcrun_log")
                next(mesg)  # skip the first entry 
                assert next(mesg)[1] == "fourty two"

    @raises(sqlite3.DatabaseError)
    def test_emptyfile(self):
        """Persistent: DB file exists, but is in a wrong format (or simply is empty)."""
        with open(self.db_fname, 'w') as f:
            f.write('foo bar')
        handle, is_fresh = db.setup_db(self.model, self.db_fname, self.seed)

    @raises(sqlite3.OperationalError)
    def test_cantwrite(self):
        """Persistent: Opening a DB w/out write permissions."""
        db.setup_db(self.model, "/foo.db", self.seed)


    @raises(RuntimeError)
    def test_wrong_model(self):
        model = MockModel()
        db.setup_db(model, self.db_fname, seed=42)
        # now restart w/ different model
        model.id_str = "gobbledegook"
        db.setup_db(model, self.db_fname, seed=42)





class TestCreateInMemory(object):
    attrib = "quick"

    db_fname = ":memory:"
    seed = 42

    def setup(self):
        self.model = MockModel()

    def test_dbl_context(self):
        """In_memory: Double context"""
        handle, is_fresh =  db.setup_db(self.model, self.db_fname, self.seed)
        with handle:
            with handle:
                handle, is_fresh = db.setup_db(self.model, self.db_fname, self.seed)
            db.get_param(handle)

    def test_log_trans(self):
        """In_memory: Check the commits are done properly."""
        handle, is_fresh = db.setup_db(self.model, self.db_fname, self.seed)
        with handle:
            with handle:
                db._log_trans(handle,"fourty two", None)
                mesg = handle.execute("SELECT * from mcrun_log")
                next(mesg)  # skip the first entry 
        assert next(mesg)[1] == "fourty two"

    def test_row_iter(self):
        """In_memory: use row_iterator """
        handle, is_fresh = db.setup_db(self.model, self.db_fname, self.seed)

        # mock the cache
        cache = mock_cache()
        db.dump_cache(handle, cache)

        assert [-42, 11, 0] == [int(row["FSfq"]) for row in
                db.row_iterator(handle, orderby="FSfq*FSfq")]
        assert [0, 11, -42] == [int(row["FSfq"]) for row in
                db.row_iterator(handle, orderby="FSfq*FSfq ASC")]
        assert [0, 11, -42] == [int(row["FSfq"]) for row in
                db.row_iterator(handle, orderby="FSfq*FSfq", order="ASC")]



class TestSchemaNoRoots(object):
    attrib = "quick"

    db_fname = ":memory:"
    seed = 42

    def setup(self):
        self.model = MockModel()

    def test_no_roots(self):
        """Test store_roots=False."""
        cache = mock_cache()
        handle, is_fresh = db.setup_db(self.model, self.db_fname, self.seed)
        db.dump_cache(handle, cache, store_roots=False)
        assert len(cache) == \
                handle.execute("SELECT count(*) FROM mcrun_data").fetchone()[0]
        assert all( row["roots"] is None for row in db.row_iterator(handle) )


#BROKEN, never worked, is worth fixing? 
@nottest
class TestMultipleRuns(object):
    attrib="quick"
    model1 = MockModel(Par(N=3))
    model2 = MockModel(Par(N=5))
    db_fname = "test_multple.sqlite"

    def setup(self):
        remove_file(self.db_fname)

    def teardown(self):
        remove_file(self.db_fname)

    def test_setup(self):
        handle1, is_fresh1 = db.setup_db(self.model1, self.db_fname, 42)
        handle2, is_fresh2 = db.setup_db(self.model2, self.db_fname, 43)

