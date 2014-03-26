"""
DB-related model-dependent helpers & primitives
"""
from __future__ import division, print_function, absolute_import

from ...db import get_param
from .ph_param import par_id_str

def get_model_name(handle):
    """Untangle the model name from the db id_string."""
    with handle:
        db_idstr, = handle.execute("""SELECT id_str 
                                      FROM mcrun_meta;""").fetchone()
        par = get_param(handle)
        par_idstr = par_id_str(par)
        
        # should be smth like N11L12V1mq42SinglePair
        assert db_idstr[2:].startswith(par_idstr)
        
        model_name = db_idstr[2:].lstrip(par_idstr)

        if model_name is u"":
            model_name = "SingleImpurity"
        
        return model_name
