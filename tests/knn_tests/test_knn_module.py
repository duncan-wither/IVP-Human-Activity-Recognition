#!/usr/bin/env python
""" test_knn_module.py
Description: Testing the kNN module
"""
# Libs
from knn import *

choice = mex_knn([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)], (4, 2), 5, sens_str='act',
                 down_sample_rate=40, pre_str='../../', verbose=False)
