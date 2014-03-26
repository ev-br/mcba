import numpy as np
from numpy.testing import raises, assert_allclose

from mcba.models.partitions_base import fsPairs, pairs_from_buckets
from mcba.models.impurity import Par, SingleImpurity, SinglePair, energy
from mcba.models.impurity._impurity import BaseModel 
from mcba.helpers import compare_cnfs

"""Unittest models."""

class TestBasicStuff(object):
    attrib = "quick"

    @raises(RuntimeError)
    def test_impur_evenN(self):
        par = Par(N=4, V=np.inf)
        model = SingleImpurity(par, BA_solver=None, tab_base=42)


    @raises(RuntimeError)
    def test_pair_oddN(self):
        par = Par(N=3, V=np.inf)
        model = SinglePair(par, BA_solver=None, tab_base=42)



class TestPastFails(object):
    """Make sure failures of the walker's solver don't reappear."""
    attrib = "quick"
   
    def test_135(self):
        """This was failing on elysium w/ tab_base = 1.05."""
        par = Par(N=135, L=405, V=1., m_q=63)
        model = SingleImpurity(par)
        
        ref_cnf = {'buckets': np.array([-68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 125]), 'c': 0.0031225045944998585, 'roots': np.array([-210.93365087, -207.79791427, -204.66232185, -201.52687853,
       -198.39158945, -195.25645996, -192.12149561, -188.98670221,
       -185.85208579, -182.71765262, -179.58340924, -176.44936247,
       -173.31551937, -170.18188733, -167.04847401, -163.91528739,
       -160.78233576, -157.64962775, -154.51717232, -151.38497877,
       -148.25305675, -145.1214163 , -141.99006778, -138.85902196,
       -135.72828995, -132.59788325, -129.46781374, -126.33809362,
       -123.2087355 , -120.07975229, -116.95115724, -113.82296389,
       -110.69518605, -107.56783777, -104.44093327, -101.3144869 ,
        -98.18851308,  -95.06302622,  -91.93804061,  -88.81357035,
        -85.68962921,  -82.56623049,  -79.44338689,  -76.32111036,
        -73.19941185,  -70.07830117,  -66.95778675,  -63.8378754 ,
        -60.71857209,  -57.59987963,  -54.48179847,  -51.3643264 ,
        -48.24745829,  -45.13118581,  -42.01549721,  -38.90037712,
        -35.78580632,  -32.67176162,  -29.5582158 ,  -26.44513755,
        -23.33249151,  -20.2202384 ,  -17.10833524,  -13.99673556,
        -10.88538984,   -7.77424587,   -4.66324928,   -1.55234412,
          1.55852659,    4.66942022,    7.78039386,   10.89150374,
         14.00280458,   17.11434907,   20.22618731,   23.33836641,
         26.45093009,   29.56391838,   32.67736745,   35.79130941,
         38.90577234,   42.0207802 ,   45.13635302,   48.25250692,
         51.36925437,   54.48660437,   57.60456267,   60.72313205,
         63.84231257,   66.96210187,   70.08249541,   73.2034867 ,
         76.32506765,   79.44722869,   82.56995909,   85.69324707,
         88.8170801 ,   91.94144496,   95.06632797,   98.19171509,
        101.31759205,  104.44394447,  107.57075792,  110.69801803,
        113.82571054,  116.95382138,  120.08233668,  123.21124284,
        126.34052656,  129.47017483,  132.60017502,  135.73051482,
        138.86118229,  141.99216586,  145.12345433,  148.25503687,
        151.38690303,  154.51904272,  157.6514462 ,  160.78410411,
        163.9170074 ,  167.0501474 ,  170.18351575,  173.31710439,
        176.45090561,  179.58491197,  182.71911633,  185.85351184,
        188.98809189,  192.12285016,  198.39287728,  201.52813468,
        204.66354738,  207.79911019,  210.93481816,  392.9514563 ]),
        'FSfq': 0.0032031057980538812, 'P': -0.00012671924611870286 
        }

        fs_pairs = pairs_from_buckets(ref_cnf["buckets"])
        cnf = model.calculate(fs_pairs)
        compare_cnfs(cnf, ref_cnf, atol=1e-8, rtol=1e-8)



    def test_405(self):
        """fsolve was failing w/tab_base=1.02 (is ok w/ 1.019)"""
        par = Par(N=405, L=405*3, V=1., m_q =135)
        model = SingleImpurity(par) #, tab_base=1.019)
        fs_pairs = fsPairs(h=[116], p=[252])

        cnf = model.calculate(fs_pairs)
        assert_allclose(cnf["c"], -0.0129443227525, atol=1e-8, rtol=1e-8)


    def test_945(self):
        """fsolve was failing for SingleImpurity(N=945, mq=231)"""
        par = Par(N=945, L=945*3, V=1., m_q=231)
        model = SingleImpurity(par) 
        fs_pairs = fsPairs(h=[242], p=[474])

        cnf = model.calculate(fs_pairs)
        assert_allclose(cnf["c"], -0.0055875381879511013, atol=1e-8, rtol=1e-8)


    def test_405pi20(self):
        """fsolve was failing for SingleImpurity(N=405, V=pi/20, mq=270)"""
        par = Par(N=405, L=405*3, V=0.05*np.pi, m_q=270)
        model = SingleImpurity(par) 
        fs_pairs = fsPairs(h=[-185, -63], p=[-241, 264])

        cnf = model.calculate(fs_pairs)
        assert_allclose(cnf["c"], -0.0069751709989859341, atol=1e-8, rtol=1e-8)


    def test_pair(self):
        """fsolve was failing for the SinglePair(N=112, mq=47)."""
        par = Par(N=112, L=3*112, V=0.3, m_q=47)
        model = SinglePair(par)
        fs_pairs = fsPairs(h=[-48, 38], p=[-152, 86])

        cnf = model.calculate(fs_pairs)
        assert_allclose(cnf["c"], -0.00626547817994123, atol=1e-8, rtol=1e-8) 


    def test_pair2(self):
        """fsolve was failing for the SinglePair(N=112, mq=38)."""
        par = Par(N=112, L=3*112, V=0.3, m_q=38)
        model = SinglePair(par)
        fs_pairs = fsPairs(h=[-48, 38], p=[-152, 86])

        cnf = model.calculate(fs_pairs)
        assert_allclose(cnf["c"], -0.00626547817994, atol=1e-8, rtol=1e-8) 


        
####################### PAIR #########################################


class TestMagicNumbersPair(object):
    attrib = "quick"
    
    def setup(self):
        #Sasha's comparisons:
        self.cnf_OG = [
           {"par": Par(N=4, L=12, V=1., m_q=0),
            "buckets" : np.array([-3, -2, -1, 2, 3]),
            "roots": np.array([-7.958598158233196, -5.464870521331528,
                     -2.651426438408997, 6.483741054952568, 9.591154063021152]),
            "FSfq" : -0.03610196425052719}, 

           {"par": Par(N=4, L=12, V=1., m_q=0),
            "buckets" : np.array([-4, -3, -2, 1, 4]),
            "roots": np.array([-9.565832475806912, -6.448248026208533, 
                    -3.340258322614572, 5.802896569803896, 13.551442254826123]),
            "FSfq" : 0.019919859330783572}, 

           {"par": Par(N=4, L=12, V=1., m_q=1),
            "buckets" : np.array([-4, -3, -1, 1, 3]),
            "roots": np.array([-9.578913152113492, -6.466335088626141,
                   -0.2908793879700262, 5.649264952658349, 10.686862676051312]),
            "FSfq" : -0.02709464234392968}, 

           {"par": Par(N=4, L=12, V=1., m_q=2),
            "buckets" : np.array([-4, -3, 0, 2, 3]),
            "roots": np.array([-9.863838277901348, -6.989843725765916,
                      0.630903822533676, 6.573294942194662, 9.649483238938927]),
            "FSfq" : -0.0003153801785023749}, 

           {"par": Par(N=4, L=12, V=1., m_q=1),
            "buckets" : np.array([-3, -2, -1, 1, 2]),
            "roots": np.array([-6.609923688948589, -3.6149194024932583,
                    -0.7814055436535035, 4.146825873366326, 6.859422761729027]),
            "FSfq" : 0.7178968433628841}, 

           {"par": Par(N=4, L=12, V=1., m_q=1),
            "buckets" : np.array([-3, -2, -1, 1, 2]),
            "roots": np.array([-6.609923688948589, -3.6149194024932583,
                   -0.7814055436535035, 4.146825873366326, 6.859422761729027]),
            "FSfq" : 0.7178968433628841}, 

        ]

       
    def test_compare_with_OG(self):
        for cnf_OG in self.cnf_OG:
            #print "\n", cnf_OG
            self._cmp_with_OG(cnf_OG)
            

    def _cmp_with_OG(self, cnf_OG):
        model = SinglePair(cnf_OG["par"])
        fs_pairs = pairs_from_buckets(cnf_OG["buckets"]) 
        cnf = model.calculate(fs_pairs)

        assert ( cnf["buckets"] == cnf_OG["buckets"] ).all()
        assert_allclose( cnf["FSfq"], cnf_OG["FSfq"], atol=1e-9, rtol=1e-8)
        assert_allclose( cnf["roots"], cnf_OG["roots"], atol=1e-9, rtol=1e-8)
        
