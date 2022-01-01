from mcba.models.impurity import Par, SingleImpurity
from mcba.walker import Walker
from mcba.enumerations import EnumWalker


def main():
    par = Par(N=3, L=11)
    model = SingleImpurity(par)

    #run MC
    walker = Walker(model, threshold=0.999, keep_rcache=True) 
    walker.walk()

    #now run a direct enumeration
    en_walker = EnumWalker(model, threshold=0.999, keep_rcache=True)
    en_walker.walk()

    # now compare the results: rcaches are available
    def report(tag, model, first, second):
        diff = dict((pairs, first[pairs]) for pairs in first if pairs not in second )
        print("\n", tag, len(diff))
        print("\t sum overlaps = ", model.sum_overlaps(diff.values()))
        print("\t momentum     = ", model.av_momt(diff.values()))

    report("picked by MC, missed by enumeration:", model, walker.rcache, en_walker.rcache)
    report("picked by enumeration, missed by MC:", model, en_walker.rcache, walker.rcache)



############
if __name__ == "__main__":
    main()
