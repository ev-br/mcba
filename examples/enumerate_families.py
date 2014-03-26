from mcba.models.impurity import Par, SingleImpurity, fsPairs
from mcba.models.impurity.partitions import fermi_sea
from mcba.enumerations import EnumWalker, gen_oneparametric


def main():
    par = Par(N=3, L=11, m_q=1)
    model = SingleImpurity(par)

    # Enumerate partitions with the 'hole' at the edge of the fermi surface
    fs_pairs0 = fsPairs(h=[fermi_sea(par.N)[-1]],p=[])
    walker = EnumWalkerPatched(model, fs_pairs0, db_fname=":memory:")
    walker.walk()

    import mcba.db as db
    for cnf in db.row_iterator(walker.db_handle):
        print cnf["partition"], cnf["c"], cnf["FSfq"]    


class EnumWalkerPatched(EnumWalker):
    """A walker for enumerating a one-parameteric family."""
    def __init__(self, model, fs_pairs0, *args, **kwargs):
        super(EnumWalkerPatched, self).__init__(model, *args, **kwargs)
        self.generator = gen_oneparametric(self.model, fs_pairs0)


if __name__=="__main__":
    main()
