"""
This defines the chierarchy of 'models'. A model is supposed to be a sort 
of a focal point:
* it holds parameters (system size, interaction, momentum etc)
* can tell is a certain partition is valid or not
* knows how to solve the BA
* knows how to calculate relevant matrix elements

Normally, a user (programmatic or interactive) only needs to instantiate an
instance of a model, and deal exclusively with it. 
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from . import BA
from . import mx
from . import partitions as pt
from . import ph_param as ph
from .helpers import sum_overlaps, av_momt


def _select_solver(par, tab_base):
    """Return a BA solver appropriate for the par."""
    if par.V == np.inf:
        print(r"\gamma = \infty")
        return BA.BASolverInfGamma(par)
    else:
        return BA.BASolverTwoStepTab(par, tab_base=tab_base) 



class BaseModel(object):
    """Base class for the models: A model encapsulates the parameters, 
    a BA solver and can calculate relevant matrix elements, and all other 
    model-dependent functionality to be presented to users 
    (walkers, DB, postprocessors etc).
    Specifically:
        * self.calculate(fs_pairs) works out BA eqs, matrix elements.
        * partition-related stuff: a_partition, is_valid(fs_pairs)
        * long name and *id_str*, as used by the DB layer 

    The BA solver can be supplied explicitly in ctor, otherwise default one
    will be used (TwoStepTab for finite V, InfGamma otherwise).

    This class is only intended to be inherited from by specific models. 
    """
    def __init__(self, par, BA_solver, tab_base):
        self.par = par
        # a quick sanity check 
        for attr in ["N", "V", "m_q"]:
            assert hasattr(par, attr)
        assert ph.check_par(par)

        #solvers
        if BA_solver is not None:
            self.BA_solver = BA_solver
        else:
            self.BA_solver = _select_solver(par, tab_base)

        self.Lieb_Wu_a = ph.Lieb_Wu_a(self.par)

        # These are to be overridden by ancestors
        self.basis_state = NotImplemented
        self.id_str = NotImplemented
        self.name = NotImplemented

        # c=\pm\infty is the same state, hence need to filter one of them.
        # (cf sum_overlaps / av_momt)
        # default is to keep c=-\infty (\all delta_t = 0)
        self.dbl_cnt_pred = lambda x: x != np.inf


    # Collect the common functionality from partitions, matrix elements, 
    # BA solvers etc

    def calculate(self, fs_pairs):
        """Given fs_pairs, return the full cnf."""
        cnf = self.BA_solver.solve(fs_pairs)
        cnf["FSfq"] = mx.overlap(cnf, self.basis_state, self.par)
        cnf["P"] = mx.fq_Pup_fq(cnf, self.par)
        return cnf


    def is_valid(self, fs_pairs):
        """Check if arg is a valid fs_pairs for the solver."""
        return pt.is_valid(fs_pairs, self.BA_solver.par)
        

    def a_partition(self):
        """Notice the use the solver's par here:
        it should look valid to the solver.
        """
        return pt.a_partition(self.BA_solver.par)


    def sum_overlaps(self, it):
        return sum_overlaps(it, self.dbl_cnt_pred)

    def av_momt(self, it):
        return av_momt(it, self.dbl_cnt_pred)

    def long_print_par(self):
        return ph.long_print(self.par)
        


class SingleImpurity(BaseModel):
    """A single impurity injected into a sea of non-interacting fermions."""
    def __init__(self, par, BA_solver=None, tab_base=1.02):
        super(SingleImpurity, self).__init__(par, BA_solver, tab_base)
        self.name = "SingleImpurity"

        # DB id string: here am using par.id_str() just for compatibility
        self.id_str = ph.par_id_str(self.par)
        
        if self.par.N%2 == 0:
            raise RuntimeError("SingleImpurity model needs N odd.")

        # reference state for the overlaps
        self.basis_state = mx.basis_FS(self.par.N)




class SinglePair(BaseModel):
    """A model for a single pair under the FL: after the 1st collision,
    an impurity creates a hole in the Fermi sea and goes under the FL itself.
    [Rather: we believe it might :-)]. A particle created in the collision
    flies away never to return, and what's left is a pair of an impurity
    and a hole --- with the same momenta.

    Compared to the SingleImpurity model, the differences are:
    * the base-state in the overlaps has a hole quenched at the
      initial momentum q
    * Lieb-Wu eqs have q=0
    """
    def __init__(self, par, BA_solver=None, tab_base=1.02):
        if BA_solver is not None:
            _solver = BA_solver
        else:
            _par = ph.Par(N=par.N, L=par.L, V=par.V, m_q=0)
            _solver = _select_solver(_par, tab_base)
        super(SinglePair, self).__init__(par, _solver, tab_base)
        self.name = "SinglePair"
        self.id_str = ph.par_id_str(self.par) + self.name

        self.dbl_cnt_pred = lambda x: x != -np.inf

        if self.par.N%2 == 1:
            raise RuntimeError("SinglePair model needs N even.")

        fs = pt.fermi_sea(self.par.N)
        if self.par.m_q not in fs:
            raise RuntimeError("SinglePair model: q>k_F w/ m_q=%s" % self.par.m_q)
        
        #make a hole:
        self.basis_state =  mx.basis_mq(self.par.N, self.par.m_q)

        # solver should have q=0, but the outwards-facing par 
        # is as provided in the parameter list
        assert self.par == par
        assert self.BA_solver.par.m_q == 0

