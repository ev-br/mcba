import numpy as np

from mcba.walker import Walker
from mcba.models.impurity import Par, SingleImpurity


def main():
    par = Par(N=3, L=11)
    model = SingleImpurity(par)

    #run MC
    walker = Walker(model, db_fname='N3mc.sqlite') 
    walker.walk()

    # now get the results from disk and only print  
    # those with large enough weight:
    print "\n***Largest weight partitions:"

    import mcba.db as db
    handle,  = db.get_handles('N3mc.sqlite')
    for cnf in db.row_iterator(handle, orderby="FSfq*FSfq"):
        if cnf["FSfq"]**2 > 0.005:
            print  "%s:  <FS|f_q> = %s, P=%s"%(cnf["partition"], cnf["FSfq"], cnf["P"])


###################
if __name__ == "__main__":
  main()

