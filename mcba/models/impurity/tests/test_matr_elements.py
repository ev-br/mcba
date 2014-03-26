import numpy as np
from numpy.testing import raises


import mcba.models.impurity.matrix_elements as mx
from mcba.models.impurity.ph_param import Par, initial_q
import mcba.models.impurity.BA as BA
import mcba.models.partitions_base as pt

""" Unittest the matrix elements."""


class test_singular_fqPfq(object):
    """singular values for <fq|P|fq>."""

    attrib = "quick"

    def setup(self):
        self.par = Par(L=29, N=3,V=1, m_q = 1)
        self.BAS = BA.AbstractBASystemSolver(self.par) 

    @raises(AssertionError)
    def test_badarg(self):
        mx.fq_Pup_fq({"roots" : [1, 2, 3]}, self.par)

    def test_singular(self):
        fs_pairs = pt.fsPairs(h=[-1], p=[2])
        roots = self.BAS.solve(fs_pairs, self.par)
        np.testing.assert_almost_equal(mx.fq_Pup_fq(roots, self.par), 
                                       initial_q(self.par) / (self.par.N+1.))

    def test_singular2(self):
        fs_pairs = pt.fsPairs(h=[-2, 0], p=[-5, 2])
        roots = self.BAS.solve(fs_pairs, self.par)
        np.testing.assert_almost_equal( mx.fq_Pup_fq(roots, self.par), 
                                        initial_q(self.par) / (self.par.N+1.))




class test_magic_values(object):
    attrib = "quick"

    def test_magic1(self):
        par = Par(L=29, N=3,V=1, m_q = 1)
        for BAS in [ BA.BASolverTwoStep(par), 
                     BA.BASolverTwoStepTab(par), 
                   ]:
            self._magic1(BAS, par)


    def _magic1(self, BAS, par):
        """MX: a couple of magic values. """
        fs_pairs = pt.fsPairs(h=[], p=[])
        roots = BAS.solve(fs_pairs)
        np.testing.assert_almost_equal(mx.fq_Pup_fq(roots, par), 0.1422397417825382)

        basis_fs = mx.basis_FS(par.N)
        overl = mx.overlap(roots, basis_fs, par)
        np.testing.assert_almost_equal(overl, -0.862266770)
        

