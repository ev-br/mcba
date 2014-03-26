import os

from distutils.core import setup
from distutils.extension import Extension
import numpy as np


# extension paths 
BA_path = os.path.join("mcba", "models", "impurity")
tab_path = os.path.join("mcba", "models", "tabulations")    

setup(name='mcba',
      version='0.1',
      description='BA MC enumerations',
      author='Evgeni Burovski',
      author_email='evgeny.burovskiy@gmail.com',
      packages=['mcba',
                'mcba.models',
                'mcba.models.impurity',
                'mcba.models.tabulations',
               ],
      # specifying the path to acpt_DBs seems to install all the tests, 
      # incl. those in mcba/models/impurity/tests etc
      # otherwise, none are installed. Go figure.
      package_data={'': ['tests/acpt_DBs/*.sqlite', 'tests/*py'],
                   },

      # BA & tabulations helpers, compiled
      ext_modules=[Extension(os.path.join(BA_path,"_BA_cyth"), 
                             [os.path.join(BA_path,"_BA_cyth.c")]),
                             
                   Extension(os.path.join(tab_path, "_spl_interp_feval"), 
                             [os.path.join(tab_path, "_spl_interp_feval.c")]), 
                  ],
      include_dirs = [np.get_include()],
     )
