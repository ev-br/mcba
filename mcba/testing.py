from __future__ import division, print_function, absolute_import

import os
import nose

import mcba

@nose.tools.nottest
def test(attr="quick", verbose=False):
    """Run tests.
       
       verbose=True corresponds to nose verbosity=1
       attrib = "quick", "acpt", "all"
    """
    path_to_mcba = os.path.abspath(os.path.dirname(mcba.__file__))
    curr_path = os.getcwd()

    # nose seems to go nuts is run from the directory just above mcba:
    if os.path.join(curr_path, 'mcba') == path_to_mcba:
        mesg = "Please exit the mcba source tree and relaunch " 
        mesg += "your interpreter from somewhere else." 
        print(mesg)
        return False

    argv = ["-w", path_to_mcba, "--all-modules" ]
    if verbose:
        argv += ["-v"]

    #sort out the test attributes
    attr_dct = {"quick": ["-a attrib=quick", "--with-doctest"],
                "acpt": ["-a attrib=acpt"], 
                "all": [""]
               }
    argv += attr_dct[attr]  

    nose.run(argv=argv)
