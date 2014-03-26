from mcba.models.impurity import Par, SingleImpurity
from mcba.walker_factory import WalkerFactory, split_tasks
from mcba.enumerations import EnumWalker
from mcba.helpers import copydict

def main():

    # the base set of parameters for the runs
    mc_dict = { "num_sweeps"      : 140,
              "steps_per_sweep" : 10,
              "checkp_sweeps"   : 140,
              "printout_sweeps" : 140,
              "seed"            : 42
            }

    #Prepare the list of runs. Notice the need to deepcopy dicts
    # (is everything not a reference, heh)
    tasks = []
    tasks.append(copydict(mc_dict, {"db_fname" : "N5.db", 
                                    "model" : SingleImpurity(Par(N=5)), 
                                    "threshold" : 0.999})
                )

    # Factory can run both MC and enumeration tasks: 
    tasks.append(copydict(mc_dict, {"db_fname" : "N3.db", 
                                    "model" : SingleImpurity(Par(N=3)), 
                                    "walker" : EnumWalker})
            )
    tasks.append(copydict(mc_dict, {"db_fname" : "N11.db", 
                                    "model" : SingleImpurity(Par(N=11, L=33)), 
                                    "threshold" : 0.9})
                )
    tasks.append(copydict(mc_dict, {"db_fname" : "N7.db", 
                                    "model" : SingleImpurity(Par(N=7, L=21)), 
                                    "threshold" : None})
                )

    # this will replicate each tasks to have thresholds = 0.9, 0.95, 0.999, 
    # and arrange the results into a round-robin, so that 
    # the lower threshold tasks run first, then higher threshold etc 
    tasks = split_tasks(tasks, [0.9, 0.95, 0.999])

    # This runs no more than three processes at a time
    factory = WalkerFactory(tasks, 3, freq=1, verbose=False)
    factory.start()

    #now outputs of the runs is in the .output files



#######################
if __name__ == "__main__":
    main()
