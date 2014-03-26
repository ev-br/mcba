from numpy.testing import assert_allclose

from mcba.walker import Walker
from mcba.models.impurity import Par, SingleImpurity
import mcba.enumerations as en
import mcba.db as db


"""Runnig an MC process & enumeration on a small-ish system, results should agree
"""


class TestMCvsEnum(object):
    attrib = "quick"

    def setup(self):
        self.mc_dict = {
            "num_sweeps"      : 100,
            "steps_per_sweep" : 50,
            "therm_sweeps"    : 0,
            "checkp_sweeps"   : 150,
            "printout_sweeps" : 150,
            "db_fname"        : None,
            "seed"            : 42,
            "keep_rcache"     : True,
            "threshold"       : 0.995 
                      }

    def test_N3(self):
        par = Par(N=3, L=11)
        model = SingleImpurity(par)

        #run MC
        walker = Walker(model, **self.mc_dict)
        walker.walk()
        #wcache = db.load_full_cache(walker.db_handle)
        wcache = walker.rcache
        wsum = model.sum_overlaps(wcache.itervalues())

        assert wsum > 0.995    # 'quick'

        # enumerate
        print "enumerate........."

        dcache = en.direct_enumerate(par, -4, 5, verbose = False)
        dsum = model.sum_overlaps(dcache.itervalues())
        assert dsum > 0.995

        #intersection of caches must coincide
        inters = set(wcache.keys()) & set(dcache.keys())
        for c in inters:
            assert_allclose(wcache[c]["P"], dcache[c]["P"], atol=1e-8, rtol=1e-8)
            assert_allclose(wcache[c]["FSfq"], dcache[c]["FSfq"], atol=1e-8, rtol=1e-8)



class TestMCvsEnumHeavy(object):
    attrib = "acpt"

    def test_N3(self):
        par = Par(N=15, L=45)
        model = SingleImpurity(par)

        #run MC
        walker = Walker(model, threshold=0.995, keep_rcache=True, seed=42) 
        walker.walk()

        wcache = walker.rcache
        wsum = model.sum_overlaps(wcache.itervalues())
        wP = model.av_momt(wcache.itervalues())
        assert wsum > 0.995

        #now run a direct enumeration
        en_walker = en.EnumWalker(model, threshold=0.995, keep_rcache=True)
        en_walker.walk()

        dcache = en_walker.rcache
        dsum = model.sum_overlaps(dcache.itervalues())
        dP = model.av_momt(dcache.itervalues())
        assert dsum > 0.995

        #intersection of the caches must coincide
        inters = set(wcache.keys()) & set(dcache.keys())
        for c in inters:
            assert_allclose(wcache[c]["P"], dcache[c]["P"], atol=1e-8, rtol=1e-8)
            assert_allclose(wcache[c]["FSfq"], dcache[c]["FSfq"], atol=1e-8, rtol=1e-8)

        # sum rules & momenta must agree
#        print "sum rule, sum momt = ", dsum - wsum, wP-dP
        assert abs(dsum - wsum) < 5e-3
        assert abs(dP - wP) < 5e-4


