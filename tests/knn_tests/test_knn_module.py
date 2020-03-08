#!/usr/bin/env python
""" test_knn_module.py
Created by slam at 11/02/2020

Description: Testing the kNN module
"""

# Default Libs
from sys import path

# 3rd Party Libs

# Custom Libs
path.append('../../')  # enables seeing the libs above
from knn import *

__author__ = "Duncan Wither"
__copyright__ = "Copyright 2020, Duncan Wither"
__credits__ = ["Duncan Wither"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Duncan Wither"
__email__ = ""
__status__ = "Prototype"

choice = mex_knn([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)], (4, 2), 5, sens_str='act',
                 down_sample_rate=100, pre_str='../../')
