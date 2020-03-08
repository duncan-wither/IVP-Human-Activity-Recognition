#!/usr/bin/env python
""" test_mc-dcnn_module.py
Created by slam at 08/03/2020

Description: Script to test the MC-DCNN module
"""

# Default Libs
import sys
# 3rd Party Libs

# Custom Libs
import mc_dcnn

print(sys.path)
mc_dcnn.dcnn([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)],[(4,2)],pre_str='../../')
