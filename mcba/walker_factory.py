from __future__ import division, print_function, absolute_import

import os
import sys
import time
import datetime
from collections import deque, namedtuple
from multiprocessing import Process

from .helpers import copydict, roundrobin
from .walker import Walker as mcWalker



"""A task is a dict to be **kwarged to a walker.
   A Proc is a pair of a Process (which runs a walker) and a 'resource' which is 
   locked while the process is running. 
"""
Proc = namedtuple("Proc", "proc res")


################## helpers ##########################

def _outp_fname(task):
    """Figure out the filename to send the output to."""
    if "db_fname" not in task or task["db_fname"] is None:
        return "%s.output" % os.getpid()
    else:
        nm = task["db_fname"].split(".")
        if len(nm) > 1:
            nm[-1] = "output"    # "foo.bar" --> "foo.output"
        else:
            nm.append("output")  # "foo" --> "foo.output"
        return ".".join(nm)



def _get_resource(task):
    return task["db_fname"]



def _is_locked(task, proc_list):
    """Check if a task's resource is locked by a process in a proc_list.
    >>> task1 = {"db_fname" : "foo.bar"}
    >>> task2 = {"db_fname" : "blah"}
    >>> proc_list = [ Proc(proc=None, res="foo.bar"), Proc(proc=None, res="foo") ]
    >>> _is_locked(task1, proc_list), _is_locked(task2, proc_list)
    (True, False)
    """
    return _get_resource(task) in [proc.res for proc in proc_list]



def pop_next(task_queue, proc_list):
    """Scan a task_queue for a next available task (i.e. the task whose resource
    is not locked by the entries of the proc_list), and pop it from the queue  
    while preserving the order of the queued tasks.
    Returns the task and modifies the task_queue *in place*.
    >>> proc_list = [Proc(proc=None, res="foo.bar"), Proc(proc=None, res="foo")]
    >>> t_queue = deque([{"db_fname" : "foo.bar"},   \
                         {"db_fname" : "fizz.bizz"}, \
                         {"db_fname" : "blah"},      \
                         {"db_fname" : "foo"},       \
                       ])
    >>> pop_next(t_queue, proc_list)
    {'db_fname': 'fizz.bizz'}
    >>> pop_next(t_queue, proc_list)
    {'db_fname': 'blah'}
    >>> print( pop_next(t_queue, proc_list) )
    None
    >>> print( t_queue == deque([{'db_fname': 'foo.bar'}, {'db_fname': 'foo'}]))
    True
    """
    pending, task = [], None
    while task is None and task_queue:
        task = task_queue.popleft()
        if _is_locked(task, proc_list):
            pending.append(task)        # push in into the stack
            task = None

    # put what's pending back to the queue
    while pending:
        task_queue.appendleft(pending.pop())

    return task



def _split_task(task, schedule):
    """Split the task into a list of subtasks w/ thresholds taken from the schedule.
    """
    assert isinstance(schedule, list)

    if "threshold" not in task or task["threshold"] is None:
        t_sch = schedule + [None]
    else:
        t_sch = [thr for thr in schedule if thr < task["threshold"]]
        t_sch += [task["threshold"]]

    tasks = [copydict(task, {"threshold": thr}) for thr in t_sch ]    
    return tasks


def split_tasks(tasks, schedule):
    """Split each of the tasks according to the schedule, and order them 
       as a round robin: if tasks = [task1, task2] and schedule=[0.1, 0.5], 
       then the output is 
       {task1, 0.1}, {task2, 0.1 }, {task1, 0.5}, {task2, 0.5}
    """
    _tasks = tasks if isinstance(tasks, list) else [tasks]

    spl = [_split_task(t, schedule) for t in _tasks]
    subtasks = [t for t in roundrobin(*spl)]
    return subtasks



class Redirecter(object):
    """Lightweight non-buffering file IO helper.
    If runs out of steam, replace with std library logging.Logger.
    """
    def __init__(self, fname):
        self.fname = fname
    def write(self, mesg):
        with open(self.fname,'a') as f:
            f.write(mesg)
            f.flush()
            #if mesg[-1] != "\n":  # no buffering / fails w/ multiprocessing.
            #    f.flush()
    def flush(self):
        with open(self.fname, 'a') as f:
            f.flush()

######################## the Factory itself #########################

class WalkerFactory(object):
    """WalkerFactory(tasks, num_proc=3, freq=15, verbose=False): 
     Farms out tasks to walkers, running no more than `num_proc` walkers at a time. 
	Joins the walkers as they finish, checking every `freq` seconds.

     * `tasks` is a list of dicts, each of which is then **kwarg-ed to walkers.
     * If a task contains the key "walker", the corresponding value is taken to 
       be a callable for creating this walker: 
       >>> from mcba.enumerations import EnumWalker as Walker
       >>> task = {"walker": "EnumWalker", "foo": "bar"}
       >>> Walker = task.pop("walker")
       >>> walker = Walker(**task)
       ...  #doctest: +SKIP
       
       If the key is not present, the walker to run defaults to the MC one. 

     Since tasks with the same resources are sequential, does not start a task
     if its resource is in use.
     See examples/run_multiproc.py for an example of usage.
    """
    def __init__(self, tasks, num_proc=3, freq=15, verbose=False):
        self.freq, self.num_proc, self.verbose = freq, num_proc, verbose
        self.task_queue, self.num_tasks = deque(tasks), len(tasks)
        self.procs = []

        #make sure tasks' walkers are sane:
        for task in tasks:
            try:
                assert hasattr(task["walker"], "walk")
            except KeyError:
                pass  # this is ok, will default to the mcWalker


    def start(self):
        for _ in range(min(self.num_proc, self.num_tasks)):
            self._start_process()

        # mainloop: once a walker's done, join it and start another one
        while self.procs:
            for p in self.procs:
                if not p.proc.is_alive(): 
                    p.proc.join()         # this one is done
                    self.procs.remove(p)
                    if self.task_queue: 
                        self._start_process()   # start a new one
            if self.freq is not None:
                time.sleep(self.freq)  # no need for throttling a CPU


    def _start_process(self):
        task = pop_next(self.task_queue, self.procs)

        if self.verbose:
            mesg = "\n@ %s: starting " % datetime.datetime.now()
            if task:
                mesg += " %s" % _get_resource(task) #task["par"])
                if "threshold" in task:
                    mesg += "\n\tw/ threshold = %s" % task["threshold"]
            else:
                mesg += " %s"%task 
            print(mesg)
            print("running: ") 
            for p in self.procs:
                print("\t", p)


        # start it up
        if task is not None:
            p = Process(target=self._run_walker, args=(task,))
            r = _get_resource(task)

            self.procs.append(Proc(proc=p, res=r))
            p.start()


    def _run_walker(self, kwargs):
        ofname = _outp_fname(kwargs)

        red = Redirecter(ofname)
        sys.stdout = sys.stderr = red

        try:
            Walker = kwargs.pop("walker")
        except KeyError:
            Walker = mcWalker    

        walker = Walker(**kwargs)
        walker.walk()



#######################
if __name__ == "__main__":
    import doctest
    doctest.testmod()

