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
mc_dcnn.dcnn([1,2,3,4,5],[1],pre_str='../../')