from numpy.testing import assert_allclose
from numpy import pi, inf, array, zeros
from scipy.optimize import brentq

from mcba.models.impurity import BA
from mcba.models.partitions_base import fsPairs
from mcba.models.impurity.ph_param import Par, Lieb_Wu_a
import mcba.helpers as hp


############################
#### A single BA equation
############################

class TestBAeq(object):
    attrib = "quick"
    
    # BA_eq(x, a, c, n)
    magic_values = [
            {"args": {"x":  1., "a": 1.5, "c": 0, "n":6}, "res": -29.132241266373807 },
            {"args": {"x": -30., "a": 1.5, "c": 0, "n":6}, "res": 16.881786069853522}, 
            {"args": {"x": 0.1, "a": 1.5, "c": 10, "n":1}, "res": 15.104255442874548}, 
            {"args": {"x": 0.1, "a": 1.5, "c": -5.105, "n":1}, "res": -0.00074455712545251629}, 
                   ]
    
    # Eq. and its derivative: pure python versions
    eqs, derivs = [BA.BA_eq], [BA.dx_dc] 

    def test_BAeq(self):
        """Single BA eq: magic values."""
        for eq in self.eqs:
            for val in self.magic_values:
                assert_allclose(eq(**val["args"]), val["res"], 
                        atol=1e-15, rtol=1e-15)



    def test_BAeq_dxdc(self):
        """Single BA eq: dx/dc vs numerical deriv."""
        
        dc = 1e-7
        
        for eq, deriv in zip(self.eqs, self.derivs):
            for val in self.magic_values:
                v = val["args"]
                a, c, n = v["a"], v["c"], v["n"]
                x0 = brentq(eq, 1e-15, pi-1e-15, args=(c, n, a))   
                x1 = brentq(eq, 1e-15, pi-1e-15, args=(c+dc, n, a)) 

                assert_allclose( (x1-x0)/dc, deriv(x0, a), 
                        atol=1e-7, rtol=1e-7 )



def test_Bethe_root_phase():
    """Bethe_root: magic values. """

    attrib = "quick"

    #### a=0 (V\to\inf) first 
    
    # _Bethe_root(n, a, c)
    magic_values0 = [      
            {"args": {"n":0, "a":0, "c":0}, "res": pi/2.},      # c=0
            {"args": {"n":1, "a":0, "c":0}, "res": pi/2.},
    ]

    for val in magic_values0:
        assert_allclose(BA.Bethe_root_phase(**val["args"]), val["res"], 
                atol=1e-8, rtol=1e-8 )


    # general a now (a=1 is general enough)
    magic_values = [
            {"args": {"n":0, "a":1, "c":-1000}, "res": 0.00099999866667019972},
            {"args": {"n":1, "a":1, "c":-1000}, "res": 3.1425895205149925 - pi},
            {"args": {"n":-2, "a":1, "c":1}, "res": -3.3667158775453863 -pi*(-2)},     
            {"args": {"n":2, "a":1 ,"c":1}, "res": 6.4641919318905936 - pi*2}, 
    ]
    for val in magic_values:
        assert_allclose(BA.Bethe_root_phase(**val["args"]), val["res"], 
                atol=1e-8, rtol=1e-8 )


    magic_values2 = [   # c\to +\inf
            {"args1": {"n":0, "a":1, "c":100}, 
             "args2": {"n":0, "a":1, "c":-100}, 
             "res": pi-0.020321549096966596 },
            {"args1": {"n":0, "a":1, "c":1000}, 
             "args2": {"n":0, "a":1, "c":-1000}, 
             "res": pi-0.002003148814056388},  
    ]
    for val in magic_values2:
        delta = BA.Bethe_root_phase(**val["args1"]) - BA.Bethe_root_phase(**val["args2"])
        assert_allclose( delta, val["res"], atol=1e-8, rtol=1e-8 )



def test_Bethe_root_limits():
    """Bethe root: limits of c\\to\pm\infty."""
    attrib="quick"

    par = Par(N=15, L=45, V=1.)
    n=0
    cc = [-40, -100, -500, -1000]

    # c\to -\inf    
    a = Lieb_Wu_a(par)
    difs = array([BA.Bethe_root_phase(c, n, a) -
                  BA.Bethe_root_x0(c, n, a) for c in cc])
    assert_allclose( difs, zeros(len(cc)), atol=1e-8, rtol=1e-8 )

    #c\to +\inf
    difs = array([BA.Bethe_root_phase(-c, n, a) -
                  BA.Bethe_root_xpi(-c, n, a) for c in cc])
    assert_allclose( difs, zeros(len(cc)), atol=1e-8, rtol=1e-8 )




class TestBASolverTabulation(object):
    attrib="quick"
    
    par = Par(N=3, V=1., L=45, m_q=1)
    solver = BA.BASolverTwoStepTab(par)

    
    def test_finer_grid(self):
        """Test tabulation: recalc @ a finer grid, compare."""
        print("")
        _vB = self.solver._Bethe_system
        grid = self.solver.cc
        a = Lieb_Wu_a(self.par)
        buckets = array([-199, 0, 113, 88])

        cc = array([f*x+(1.-f)*y for x,y in hp.pairwise(grid) 
                for f in [0., 1e-14, 0.3, 0.5, 0.7, 1.-1e-14]])

        for c in cc: 
            xt1 = _vB(c, buckets)
            xb1 = array( [brentq(BA.BA_eq, 1e-15, pi-1e-15, args=(c, n, a)) 
                    for n in buckets] )
            assert_allclose(xt1, xb1, atol=1e-8, rtol=1e-8)






###########################
### BA system solvers
###########################

def test_c_inf():
    """c = \pm\inf guards @AbstractBASystemSolver."""

    attrib = "quick"

    par = Par(N=3)
    Solv = BA.AbstractBASystemSolver(par)

    rts = Solv.solve( fsPairs([1], [4]), par)  #[-2, -1, 0, 4]
    assert_allclose( rts["roots"], pi * array([-2, -1, 0, 4]), 
            rtol=0, atol=1e-12 )
    assert rts["c"] == -inf

    rts = Solv.solve( fsPairs([-2, 1], [-4, 2]), par)  #[-4, -1, 0, 2]
    assert_allclose( rts["roots"], pi * array([-3, 0, 1, 3]), 
            rtol=0, atol=1e-12 )
    assert rts["c"] == inf



class test_BA_system_solver(object):
    attrib = "quick"

    par = Par(N=3)

    solvers = [ BA.BASolverTwoStep(par), 
                BA.BASolverTwoStepTab(par),
    ]


    def test_magic_values(self):
        """BA system: magic values. (loop over solvers)"""
        print("")
        for solver in self.solvers:
            print("solver = ", solver)
            self.magic_values(solver)



    def magic_values(self, BAS):
        fs_pairs = fsPairs([], [])   #[-2, -1, 0, 1]
        rts = BAS.solve(fs_pairs)
        assert_allclose( rts["roots"], [-3.46018061, -0.47462842, 
                2.31997086,  4.75643082], rtol=1e-8, atol=1e-8 )
        assert_allclose( BAS._Bethe_system_last(rts["c"], rts["buckets"]), 
                0., rtol=0, atol=1e-11 )

        fs_pairs = fsPairs([1], [2]) #[-2, -1, 0, 2]
        rts = BAS.solve(fs_pairs)
        assert_allclose( rts["roots"], [-3.73113373, -1.08209286,  
                1.26085886,  6.69396038], rtol=1e-8, atol=1e-8 )
        assert_allclose( BAS._Bethe_system_last(rts["c"], rts["buckets"]), 
                0., rtol=0, atol=1e-11 )
        #TODO: mo'magic values


    def test_find_roots(self):
        """find_roots should be  consistent w/ solve."""
        fs_pairs = fsPairs([1],[2])
        for BAS in self.solvers:
            roots = BAS.solve(fs_pairs)
            roots1 = BAS.find_roots(fs_pairs, roots["c"])
            assert_allclose( roots["roots"], roots1["roots"], 
                    atol=1e-8, rtol=1e-8 )
            assert_allclose( roots["c"], roots1["c"], 
                    atol=1e-8, rtol=1e-8 )
            assert all(roots["buckets"] == roots1["buckets"])



class TestSolverPastFails(object):
    def test_405(self):
        """Was failing w/tab_base=1.02, ok w/tab_base=1.019 """
        par = Par(N=405, L=405*3, V=1., m_q =135)
        fs_pairs = fsPairs(h=[116], p=[252])
#        model = SingleImpurity(par) #, tab_base=1.019)
        BAS_brent = BA.BASolverTwoStep(par)
        BAS_tab = BA.BASolverTwoStepTab(par)
        
        roots_brent = BAS_brent.solve(fs_pairs)
        roots_tab = BAS_tab.solve(fs_pairs)
        
        assert_allclose(roots_brent["c"], roots_tab["c"], rtol=1e-8, atol=1e-8)
        
        

