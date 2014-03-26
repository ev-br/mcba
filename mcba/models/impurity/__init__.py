"""
This is supposed to be a single stop shop for all the user code.

A user (programmatic or interactive) only needs to import from this package
directly, and treat all other the individual modules as implementation details.

"""
from __future__ import division, print_function, absolute_import

from . import BA
from . import matrix_elements as mx
from .matrix_elements import energy
from .ph_param import Par, gamma, initial_q, k_F, E_F, E_FS, E_in
from .partitions import fsPairs
from .helpers import sum_overlaps, av_momt
from ._impurity import SingleImpurity, SinglePair

