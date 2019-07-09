from numpy.testing import raises
from numpy import inf
import os


import mcba.enumerations as en
from mcba.models.impurity import Par, SingleImpurity, SinglePair
from mcba.models.partitions_base import fsPairs
from mcba.helpers import copydict


class TestThreadHoles(object):
    attrib = "quick"
    par = Par(N=3, m_q=1)
    model = SingleImpurity(par)

    @raises(AssertionError)
    def test_badarg(self):
        """Particles are outside of the FS."""
        self._generate_pairs([0])

    def test_num1(self):
        assert self._generate_pairs([5]) == []

    def test_num2(self):
        assert self._generate_pairs([4]) == [ fsPairs(h=[1], p=[4]) ]

    def test_num3(self):
        assert self._generate_pairs([3]) == \
                [ fsPairs(h=[0], p=[3]), fsPairs(h=[1], p=[3]) ]


    def _generate_pairs(self, particles):
        """Wrap thread_holes() into a list."""
        return [pairs for pairs in en.thread_holes(particles, self.model)]


class TestGenSupport(object):
    attrib = "quick"

    def test_stop_none(self):
        """If num_stop=None, gen_support goes on indefinitely."""
        cnt = 0
        for cnt, (x, s) in enumerate(en.gen_support(3, 0, None)):
            if x>7:
                break
        print(x, cnt)
        assert cnt==12 and x==8 



class TestGenPartitionsBase(object):
    attrib = "dear_nose_this_is_not_a_test"

    def setup(self):
        self.par = NotImplemented
        self.model = NotImplemented

    def _generate_partitions_1(self):
        """Wrap gen_partitions into a list."""
        return [pairs for pairs in en.gen_partitions_1(self.model)]

    def _generate_partitions(self, num_p, num_start, num_stop=None):
        return [pairs for pairs in en.gen_partitions(num_p, self.model, num_start, num_stop)]

#    @raises(AssertionError)
#    def test_badarg(self):
#        self._generate_partitions_1(-1)

    def test_zero(self):
        assert self._generate_partitions(0, 0) == \
                [(0, fsPairs(h=[], p=[]))]   # FS=[-2, -1, 0, 1]




class TestGenPartitionsImpur(TestGenPartitionsBase):
    """Test generate_partitions: SingleImpurity model."""
    attrib="quick"
    def setup(self):
        self.par = Par(N=3, m_q=1)
        self.model = SingleImpurity(self.par)

    def test_one(self):
        assert self._generate_partitions_1() == \
                [(-3, fsPairs(h=[-2], p=[-3])),   # [-3, -1, 0, 1]
                 (2, fsPairs(h=[-1], p=[2])),     # [-2,  0, 1, 2]
                 (2, fsPairs(h=[0], p=[2])),      # [-2, -1, 1, 2]
                 (2, fsPairs(h=[1], p=[2])),      # [-2, -1, 0, 2]
                 (3, fsPairs(h=[0], p=[3])),      # [-2, -1, 1, 3]
                 (3, fsPairs(h=[1], p=[3])),      # [-2, -1, 0, 3]
                 (4, fsPairs(h=[1], p=[4]))]      # [-2, -1, 0, 4]


    def test_two(self):
        res = self._generate_partitions(2, 0, 3)
        assert res == \
                [(-3, fsPairs(h=[-2, -1], p=[-3, 2])), 
                 (-3, fsPairs(h=[-2, 0], p=[-3, 2])), 
                 (-3, fsPairs(h=[-2, 1], p=[-3, 2])), 
                 (-3, fsPairs(h=[-1, 0], p=[-3, 2])), 
                 (-3, fsPairs(h=[-1, 1], p=[-3, 2])), 
                 (3, fsPairs(h=[-2, -1], p=[3, -3])), 
                 (3, fsPairs(h=[-2, 0], p=[3, -3])), 
                 (3, fsPairs(h=[-2, 1], p=[3, -3])), 
                 (3, fsPairs(h=[-1, 0], p=[3, -3])), 
                 (3, fsPairs(h=[-1, 1], p=[3, -3])), 
                 (3, fsPairs(h=[0, 1], p=[3, -3]))]

    def test_two_restart(self):
        r1 = self._generate_partitions(2, 0, 3)
        r2 = self._generate_partitions(2, 3, 5)
        assert r1+r2 == self._generate_partitions(2, 0, 5)



class TestGenPartitionsPair(TestGenPartitionsBase):
    """Test generate_partitions: SinglePair model."""
    attrib="quick"
    def setup(self):
        self.par = Par(N=4, m_q=1)
        self.model = SinglePair(self.par)

    def test_one(self):
        assert self._generate_partitions_1() ==\
                [(-4, fsPairs(h=[-2], p=[-4])),
                (-4, fsPairs(h=[-1], p=[-4])),
                (-4, fsPairs(h=[0], p=[-4])),
                (-4, fsPairs(h=[1], p=[-4])),
                (-3, fsPairs(h=[-2], p=[-3])),
                (-3, fsPairs(h=[-1], p=[-3])),
                (-3, fsPairs(h=[0], p=[-3])),
                (-3, fsPairs(h=[1], p=[-3])),
                (-3, fsPairs(h=[2], p=[-3]))]


    def test_two(self):
        res = self._generate_partitions(2, 0, 3)
        assert res == \
                [(3, fsPairs(h=[-2, 2], p=[3, -3])),
                (3, fsPairs(h=[-1, 1], p=[3, -3])),
                (3, fsPairs(h=[-1, 2], p=[3, -3])),
                (3, fsPairs(h=[0, 1], p=[3, -3])),
                (3, fsPairs(h=[0, 2], p=[3, -3])),
                (3, fsPairs(h=[1, 2], p=[3, -3])),
                (-4, fsPairs(h=[-2, -1], p=[-4, -3])),
                (-4, fsPairs(h=[-2, 0], p=[-4, -3])),
                (-4, fsPairs(h=[-2, 1], p=[-4, 3])),
                (-4, fsPairs(h=[-2, 2], p=[-4, 3])),
                (-4, fsPairs(h=[-1, 0], p=[-4, 3])),
                (-4, fsPairs(h=[-1, 1], p=[-4, 3])),
                (-4, fsPairs(h=[-1, 2], p=[-4, 3])),
                (-4, fsPairs(h=[0, 1], p=[-4, 3])),
                (-4, fsPairs(h=[0, 2], p=[-4, 3])),
                (-4, fsPairs(h=[1, 2], p=[-4, 3]))]

    def test_two_restart(self):
        r1 = self._generate_partitions(2, 0, 3)
        r2 = self._generate_partitions(2, 3, 5)
        assert r1+r2 == self._generate_partitions(2, 0, 5)



def remove_file(fname):
    """A helper for setup/teardown.""" 
    if os.path.exists(fname):
        os.remove(fname)

class TestEnumWalker(object):
    attrib="quick"
    par = Par(N=3, m_q=1, V=inf)
    model = SingleImpurity(par)
    en_dict = {"max_num_p": 2, 
               "keep_rcache": True,
               "db_fname": "dummytest_n3.db", }
    
    def setup(self):
        remove_file(self.en_dict["db_fname"])

    def teardown(self):
        remove_file(self.en_dict["db_fname"])
    
    def test_walker_restart(self):
        # walk
        walker1 = en.EnumWalker(self.model, max_limit=3,  **self.en_dict)
        walker1.walk()

        # keep walking
        walker2 = en.EnumWalker(self.model, max_limit=6, **self.en_dict)
        walker2.walk()

        #mix the cnfs together
        cache12 = walker1.rcache
        cache12.update(walker2.rcache)

        # now start afresh
        en_dict = copydict(self.en_dict,{"db_fname": ":memory:"})
        walker3 = en.EnumWalker(self.model, max_limit=6, **en_dict)
        walker3.walk()
    
        #make sure nothing's lost
        assert len(cache12) == len(walker3.rcache)
        for a,b in zip(cache12, walker3.rcache):
            assert a in walker3.rcache
            assert b in cache12

