import os

from nose.tools import raises #, nottest
from numpy.testing import assert_allclose
from numpy import inf

from mcba.walker_factory import WalkerFactory, _split_task, split_tasks
from mcba.models.impurity import Par, SingleImpurity
import mcba.db as db
from mcba.helpers import copydict


class TestSplitTasks(object):
    attrib = "quick"

    def test_split_one_task(self):
        schedule = [0.9, 0.95, 0.999]

        spl = _split_task({"threshold": 0.995}, schedule)

        # should be [{'threshold': 0.9}, {'threshold': 0.95}, {'threshold': 0.995}]
        assert len(spl)==3
        assert_allclose( [s["threshold"] for s in spl], [0.9, 0.95, 0.995])


        spl = _split_task({"threshold": None}, schedule)
        # should be [{'threshold': 0.9}, {'threshold': 0.95}, {'threshold': 0.999}, {'threshold': None}]

        assert len(spl)==4
        assert_allclose( [s["threshold"] for s in spl[:-1]], [0.9, 0.95, 0.999])
        assert spl[-1]["threshold"] is None


        spl = _split_task({"a": 42}, [0.9, 0.95])
        #should be [{'a': 42, 'threshold': 0.9}, {'a': 42, 'threshold': 0.95}, {'a': 42, 'threshold': None}]
        
        assert len(spl)==3
        assert_allclose( [s["threshold"] for s in spl[:-1]], [0.9, 0.95])
        assert spl[-1]["threshold"] is None


    def test_split_tasks(self):
        schedule = [0.25, 0.5, 1.0]
        tasks = [{'a': 42, 'threshold': 0.75}, {'a': 43, 'threshold': 0.75}]
        
        subtasks = split_tasks(tasks, schedule)

        assert_allclose( [ t["threshold"] for t in subtasks], [0.25, 0.25, 0.5, 0.5, 0.75, 0.75] )
        assert [t["a"] for t in subtasks] == [42, 43, 42, 43, 42, 43]


    def test_split_tasks_single(self):
        schedule = [0.25, 0.5, 1.0]
        tasks = {'a': 42, 'threshold': 0.75}
        
        subtasks = split_tasks(tasks, schedule)

        assert_allclose( [ t["threshold"] for t in subtasks], [0.25, 0.5, 0.75] )
        assert [t["a"] for t in subtasks] == [42, 42, 42]




class TestFactory(object):
    attrib = "quick"

    def setup(self):
        self.mc_dict = {
            "num_sweeps"      : 1,
            "steps_per_sweep" : 10,
            "therm_sweeps"    : 0,
            "checkp_sweeps"   : 20,
            "printout_sweeps" : 20,
            "low_en_range"    : 1,
            "seed"            : 42
                       }

        self.tasks = [copydict(self.mc_dict, 
                {"db_fname" : "N3.db", 
                 "model": SingleImpurity(Par(N=3, V=inf))}),
                     ]

        self.tasks.append(copydict(self.mc_dict, 
                {"db_fname" : "N5.db", #"threshold" : 0.99, 
                 "model" : SingleImpurity(Par(N=5, V=inf)),}
                         ))


    def teardown(self):
        for task in self.tasks:
            if os.path.exists(task["db_fname"]):
                os.remove(task["db_fname"])
        for task in self.tasks:
            of = task["db_fname"].split(".")
            of[-1] = "output"
            ofname = ".".join(of)
            if os.path.exists(ofname):
                os.remove(ofname)


    def test_walk(self):
        factory = WalkerFactory(self.tasks, 2, freq=0.05)
        factory.start()

        for task in self.tasks:
           # print "\n\n**********", task["db_fname"]
            assert os.path.exists(task["db_fname"])


    @raises(AssertionError)
    def test_not_a_walker(self):
        task = [{"walker": "foo", "arg": "bar"}]

        factory = WalkerFactory(task, 1)
        factory = WalkerFactory(self.tasks, 2, freq=1)

