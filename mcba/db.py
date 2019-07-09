from __future__ import division, print_function, absolute_import
import sqlite3
import datetime
import os
import zlib
import numpy
import warnings
from collections import namedtuple

from .models.partitions_base import pre_hash, from_prehash, pairs_from_buckets

#verify_DB only:
import random    # NB: only used @ verify_DB; do NOT use anywhere else, 
                 # use numpy.random instead
from .helpers import  compare_cnfs 
from .models.partitions_base import fsPairs 


#TODO: abstractions leak: models & roots
# First, roots=None should mean that the schema does not have the roots column.  
# Second, a model should define the fields for the _meta table (N,L,V,q etc), 
#  and the ctor for the parameters (or is a namedtuple good enough?)
# Third, _results table is model-dependent as well


############# Adapters/converters for the DB #####################


######## this is for the buckets, deprecated
def adapt_list(lst):
    return buffer(zlib.compress( " ".join(str(n) for n in lst) ))

def convert_buckets(s):
    return [int(n) for n in zlib.decompress(s).split()]


######## fs_pairs
def adapt_tuple(tpl):
    # cf http://stackoverflow.com/questions/15055189/
    # esp unutbu's answer
    return " ".join(str(n) for n in tpl) or " "

def convert_fs_pairs(s):
    t = tuple([int(x) for x in s.split()])
    fsp = from_prehash(t)
    return fsp


""" NB: to/fromstring goes nuts if the dtypes are involved
>>> np.fromstring( np.array([42]).tostring() )
array([  2.07507571e-322])
"""
def adapt_ndarray(ndarr):
    return buffer(zlib.compress(ndarr.tostring()))

def convert_ndarray(string):
    return numpy.fromstring(zlib.decompress(string))


class DBHandle(object):
    """Wrap an sqlite3.connection object: re-open/close context for a transaction.
    For an in-memory DB, keep it open.

    For persistent DBs keeps the count of the #of connections, closes the DB
    once the count goes down to 0. It should thus be safe to nest contexts, 
    for example:
        with db_handle:
            with db_handle:
                setup_db(...)
                dump_cache(...)
    """
    def __init__(self, fname):
        """If an in-memory DB, connect immediately. Otherwise, do nothing."""
        self.fname = fname
        self.is_persistent = (fname != ":memory:")
        self.count = 0    # number of _opens

        sqlite3.register_adapter(tuple, adapt_tuple)
        sqlite3.register_converter("tuple_of_ints", convert_fs_pairs)

        #DEPRECATE: buckets as a partition
        sqlite3.register_adapter(list, adapt_list)
        sqlite3.register_converter("list_of_ints", convert_buckets)

        sqlite3.register_adapter(numpy.ndarray, adapt_ndarray)
        sqlite3.register_converter("ndarray", convert_ndarray)

        if not self.is_persistent:
            self.conn, self.count = self._open(self.fname)
        else:
            self.conn, self.count = None, 0

    def _open(self, fname):
        self.count += 1
        if self.count == 1:
            self.conn = sqlite3.connect(fname, 
                    detect_types=sqlite3.PARSE_DECLTYPES)
            self.conn.row_factory = sqlite3.Row
        return self.conn, self.count

    def _close(self):
        assert self.count > 0
        self.count -= 1
        self.conn.commit()
        if self.count == 0:
            self.conn.close()
        return self.conn, self.count

    def __enter__(self):
        if self.is_persistent:
            self.conn, self.count = self._open(self.fname)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_persistent:
            self.conn, self.count = self._close()
        return False     # not gonna handle errors, hence False

    ############# these should not be called out of context #################
    def execute(self, *args):
        assert self.is_usable()
        return self.conn.execute(*args)

    def commit(self):
        self.conn.commit()
        return self

    # subclass and override, if desired
    def is_usable(self):
        return self.conn is not None and self.count > 0



class DBHandleWithID(DBHandle):
    """DB handle + run id_str :-)."""
    def __init__(self, fname, id_str=None, db_key=None):
        super(DBHandleWithID, self).__init__(fname)
        self.id_str = id_str
        self.db_key = db_key

    def is_key_ok(self):
        """Check if the db_key stored equals that fetched."""
        res = self.db_key == self.execute("""SELECT db_key FROM mcrun_meta
                WHERE id_str=(?)""", (self.id_str,)).fetchone()[0]
        return res

    def is_usable(self):
        return (super(DBHandleWithID, self).is_usable() and 
                self.id_str is not None)
        # NB: can't check if is_key_ok here: execute() creates an inf recursion


##############################################################################
########################## User-callables ####################################
##############################################################################

def get_handles(fname):
    """Given a fname, return the list of DB handles."""
    # if file does not exist, let the exception propagate (it's a failure anyway)
    conn = sqlite3.connect(fname)
    handles = [DBHandleWithID(fname, id_str, db_key) for (id_str, db_key)
            in conn.execute("SELECT id_str, db_key FROM mcrun_meta;")]
    conn.close()
    return handles


def setup_db(model, db_fname, seed, prefix='mc'):
    """Setup the DB and return the handle: if new, create one; if existing, 
    check compatibility.
    """
    id_str = prefix+model.id_str
    schema_tag = "foo"

    if db_fname is None:
        handle = DBHandleWithID(":memory:", id_str)
    else:
        handle = DBHandleWithID(db_fname, id_str)

    with handle:
        """if existing db, check that parameters are same
           otherwise, set it up
        """
        try:
            # this throws if table doesn't exist; 
            # otherwise returns a (key, schema_tag)
            # --- or None in case of a wrong schema
            db_key, db_sch = handle.execute("""SELECT db_key, schema
                    FROM mcrun_meta WHERE id_str=(?)""", (id_str,)).fetchone()

            # TODO: the table exists, but the key's not there. Need to generate 
            # the next key etc. Is this actually worth it? 
            if db_key is None:
                mesg = "DB: no db_key for id_str=%s in %s" % (id_str, db_fname)
                raise NotImplementedError(mesg)
            else:
                handle.db_key = db_key

            # being paranoid: make sure the parameters are same
            par1 = get_param(handle)
            if model.par != par1:
                raise RuntimeError("DB: %s : par does not match." % (db_fname,))

            #make sure the schema is the same
            if db_sch != schema_tag:
                mesg = "DB: schema %s does not match %s" % (db_sch, schema_tag)
                raise RuntimeError(mesg)

            _log_trans(handle, "Re-setup: restarting? pid= %s" % os.getpid(), 
                        None)
            is_fresh = False

        except sqlite3.OperationalError:  # empty db, fill the blanks in
            handle.db_key = 0
            par = model.par
            meta = (id_str, par.L, par.N, par.V, par.m_q, seed, 
                    handle.db_key, schema_tag)
            _create_tables(handle)
            handle.execute("INSERT INTO mcrun_meta VALUES(?,?,?,?,?,?,?, ?);", 
                            meta)
            handle.execute("""INSERT INTO mcrun_results
                         VALUES(?,?,?,?);""", (handle.db_key, 0., 0., "init") )
            _log_trans(handle, "Starting afresh. pid= %s"%os.getpid(), None)
            handle.commit()
            is_fresh = True

        except TypeError:   
            """Is being thrown if unpacking None into (db_key, db_sch):  
               the table exists, but there's no appropriate id_str: wrong schema?
            """
            mesg = "DB: no match for %s in the DB" % id_str
            raise RuntimeError(mesg)
        assert handle.is_usable() and handle.is_key_ok()
    return handle, is_fresh


def dump_cache(handle, rcache, store_roots=True):
    """ Dumps the cache to the DB.
    Relies on the DB schema for partitions being unique.
    """
    with handle:
        assert handle.is_usable() and handle.is_key_ok()
        for pairs in rcache:
            try:
                entry = (handle.db_key, pairs, 
                         (rcache[pairs]["roots"] if store_roots else None),
                         rcache[pairs]["c"], rcache[pairs]["FSfq"], 

                         rcache[pairs]["P"])
                handle.execute("""INSERT INTO mcrun_data
                                  VALUES (?, ?, ?, ?, ?, ?);""", entry)
            except sqlite3.IntegrityError:
                pass    # partitions must be unique
        handle.commit()
'''
#
# An alternative version: Builds a set of the partitions already in DB, 
# dumps a *compement* of rcache only. 
# May get *very* memory-intensive
#
    """
    with handle:
        db_partitions = handle.execute("""SELECT partition from mcrun_data
                                          WHERE db_key=?""", (handle.db_key,))

        db_set = set( pre_hash(x[0]) for x in db_partitions)
        diff = set(rcache.keys()) - db_set

        todump = [ (handle.db_key, p, #rcache[p]["buckets"], 
                rcache[p]["roots"],
                rcache[p]["c"], rcache[p]["FSfq"], rcache[p]["P"] )
                for p in diff ]

        for entry in todump:
            handle.execute("""INSERT INTO mcrun_data
                              VALUES (?, ?, ?, ?, ?, ?);""", entry)
        handle.commit()
'''

def update_results(handle, sum_overl, sum_p, status):
    """Update the results table."""
    with handle:
        assert handle.is_usable() and handle.is_key_ok()
        t = (sum_overl, sum_p, status, handle.db_key)
        handle.execute("""UPDATE mcrun_results SET sumFSfq2=?, sumP=?, status=?
                          WHERE db_key=?""", t)


def row_iterator(handle, orderby=None, order="DESC"):
    """Open a handle & return an iterator over the _data table rows."""
    with handle:
        assert handle.is_usable() and handle.is_key_ok()
        sql_stmt = "SELECT * FROM mcrun_data WHERE db_key=%s"%handle.db_key

        if orderby:
            sql_stmt += " ORDER BY %s"%orderby
            if not any(x in sql_stmt for x in ["ASC", "DESC"]):
                sql_stmt += " %s"%order

        rows = handle.execute(sql_stmt)
        for row in rows:
            yield row
'''
Imposing an order does seem to slow things down. Here's an extreme example (1187728 rows)

$ python -mtimeit -s"import mcba.db as db; handle, = db.get_handles('../data/inf_gamma/N45ginfmq18.en.sqlite')" "for row in db.row_iterator(handle): pass"
10 loops, best of 3: 16.1 sec per loop
br@ymir:$ 
$ python -mtimeit -s"import mcba.db as db; handle, = db.get_handles('../data/inf_gamma/N45ginfmq18.en.sqlite')" "for row in db.row_iterator(handle, orderby='FSfq*FSfq'): pass"
10 loops, best of 3: 55.5 sec per loop

For smaller DBs the difference is negligible (../data/q25/N5g6mq4.sqlite, 811 rows):
w/o ordering: 100 loops, best of 3: 10.2 msec per loop
w/ ordering:  100 loops, best of 3: 12.1 msec per loop

A medium-sized DB (../data/q25/N45g6mq6.sqlite, 109545 rows) a factor of three again:
w/o ordering:  10 loops, best of 3: 1.47 sec per loop
w/ ordering:   10 loops, best of 3: 4.17 sec per loop
'''

def get_param(handle):
    """Construct and return a Par() object from a DB handle.
    TODO: In fact, this returns a namedtuple with the fields.
    """
    pseudopar = namedtuple("Par", "N L V m_q")
    
    with handle:
        assert handle.is_usable() and handle.is_key_ok()
        x = handle.execute("SELECT * FROM mcrun_meta WHERE id_str=?",
                 (handle.id_str,)).fetchone()
    return pseudopar(N=x[2], L=x[1], V=x[3], m_q=x[4])


def get_db_key(handle):
    with handle:
        assert handle.is_usable() and handle.is_key_ok()
        db_key = handle.execute("""SELECT db_key FROM mcrun_meta
                WHERE id_str=(?)""", (handle.id_str,)).fetchone()[0]
    return db_key


def get_results(handle):
    with handle:
        assert handle.is_usable() and handle.is_key_ok()
        cursor = handle.execute("""SELECT * FROM mcrun_results
                                   WHERE db_key=(?)""", (handle.db_key,))
        res = cursor.fetchone()
    return { "sum FSfq2" : res[1], "sum P" : res[2], "status" : res[3] }


def get_DBsize(handle):
    with handle:
        size = handle.execute("""SELECT count(*) FROM mcrun_data;""").fetchone()
    return size[0]


def verify_DB(handle, model, num, check_roots=True, atol=1e-10, rtol=1e-10):
    """Recalculate num entries of the DB, chosen at random. 
    If num=None, recalculate all entries (SLOW). 
    """
    with handle:
        assert handle.is_usable() and handle.is_key_ok()
        print("reading the DB...")

        par = get_param(handle)
        assert par == model.par
        
        it =  row_iterator(handle)
        if num is not None:
            # select num rows from the DB
            dbsize = get_DBsize(handle)
            try:
                indices = random.sample(range(dbsize), num)
                it = (x for j, x in enumerate(it) if j in indices)
            except ValueError:
                pass         # "random.sample: sample is larger than population"
        print("...done")

        for x in it:
            fs_pairs = x["partition"]
            if x['partition'] is None:
                # that py2.6 thing w/ Nones for empty tuples
                warnings.warn('Found None in the DB. Interpreting it as FS.', 
                        DeprecationWarning)
                fs_pairs = fsPairs(h=[], p=[])
            elif not isinstance(x["partition"], fsPairs): 
                # this must be buckets, convert to fs_pairs
                warnings.warn('''Found somethign in the DB. 
                              Interpreting as buckets''',  DeprecationWarning)
                fs_pairs = pairs_from_buckets(x["partition"])

            print("checking ", fs_pairs, " ...", end='')
            cnf = model.calculate(fs_pairs)
            compare_cnfs(cnf, x, check_roots, atol, rtol)
            print("ok")
    return True



########################### helpers #################################

def _create_tables(handle):
    handle.execute("""CREATE TABLE mcrun_meta
                   (id_str   text   not null unique,
                    L        int    not null,
                    N        int    not null,
                    V        double not null,
                    m_q      int    not null,
                    seed     int,
                    db_key   int    not null unique, 
                    schema   text
                   );""")
    handle.execute("""CREATE TABLE mcrun_data
                   (db_key     int           not null,
                    partition  tuple_of_ints  not null unique,
                    roots      ndarray,
                    c          double        not null,
                    FSfq       double        not null,
                    P          double        not null
                   );""")
    handle.execute("""CREATE TABLE mcrun_results
                   (db_key     int   not null unique,
                    sumFSfq2   double,
                    sumP       double,
                    status     text
                   );""")
    handle.execute("""CREATE TABLE mcrun_log
                   (db_key     int   not null,
                    date       text,
                    time       text,
                    status     text,
                    details    text
                   );""")
    handle.commit()


def _log_trans(handle, msg, extra_msg=None):
    datestr, timestr = ("%s"%datetime.datetime.now()).split()
    with handle:
        handle.execute("""INSERT INTO mcrun_log VALUES(?,?,?,?,?);""",
                             (handle.db_key, msg, datestr, timestr, extra_msg))

