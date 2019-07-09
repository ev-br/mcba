import numpy as np

from mcba.models.tabulations.piecewise_cubic import PiecewiseCubic

def parab(x):
    return x**2
def d_parab(x):
    return 2.*x

def fill_above(x):
    return -42.
    
def fill_below(x):
    return -43.
    

class test_tab_parabola(object):
    attrib="quick"

    ntab = 50
    xmin, xmax = 0., 5.
    xx = np.array([xmin + (xmax-xmin)*j/ntab for j in range(ntab+1)])
    tab = PiecewiseCubic(xx, parab, d_parab, 
            f_above=fill_above, f_below=fill_below)


    def test_inside(self):
        """own-tabulate a parabola."""
        ntab1 = 888*self.ntab
        xmin1, xmax1 = self.xmin, self.xmax
        xt = np.array([ xmin1 + (xmax1-xmin1)*j/ntab1 for j in range(ntab1) ])

        yp = np.array([parab(x) for x in xt])
        yt = self.tab(xt)

        np.testing.assert_allclose(yt, yp, atol=1e-10, rtol=1e-10)


    def test_outside(self):
        """tabulation: above/below the range."""
        x = self.xmin-self.xmax
        np.testing.assert_allclose( self.tab(x), fill_below(x) )

        x = 2.*self.xmax
        np.testing.assert_allclose( self.tab(x), fill_above(x) )




