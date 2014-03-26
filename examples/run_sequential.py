import os
import numpy as np

from mcba.walker_factory import split_tasks
from mcba.walker import Walker
from mcba.models.impurity import Par, SingleImpurity
from mcba.helpers import copydict

def main():

    # the base set of parameters for the runs
    mc_dict = { "num_sweeps"      : np.inf,
              "steps_per_sweep" : 100,
              "checkp_sweeps"   : 100,
              "printout_sweeps" : 100,
              "seed"            : 42,
            }
    N=5
    L=3*N
    mqs = [1, 2, 3, 4, 5]
    base_path = "."

    #Prepare the list of runs. Notice the need to deepcopy dicts
    # (is everything not a reference, heh)
    tasks = [ copydict(mc_dict, {"model" : SingleImpurity(Par(N=N, L=L,
                                                              V=0.2, m_q=m_q)),
                                 "db_fname" : os.path.join(base_path,
                                 "N{0}g1.2mq{1}.sqlite".format(N, m_q))})
            for m_q in mqs ]


    # this will replicate each tasks to have thresholds = 0.9, 0.95, 0.999,
    # and arrange the results into a round-robin, so that
    # all the lower threshold tasks run first, then higher threshold etc
    tasks = split_tasks(tasks, [0.9, 0.95, 0.999])

    for task in tasks:
        print "starting ", task["model"].par
        walker = Walker(**task)
        walker.walk()



#######################
if __name__ == "__main__":
    main()
