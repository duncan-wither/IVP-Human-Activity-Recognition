#!/usr/bin/env python
""" Mex_utils.py
Created by slam at 12/03/2020

Description: Some utility functions for the MEX Project.
"""

# Default Libs

# 3rd Party Libs
import numpy as np

# Custom Libs

__author__ = "Duncan Wither"
__copyright__ = "Copyright 2020, Duncan Wither"
__credits__ = ["Duncan Wither"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Duncan Wither"
__email__ = ""
__status__ = "Prototype"


def create_mex_str(patient_no, exercise_no, sensor_str, pre_str=""):
    str_1 = pre_str + 'MEx Dataset/Dataset'
    
    # Setting up which exercise to use
    if sensor_str == 'act':
        sens_str1 = '/act'
    elif sensor_str == 'acw':
        sens_str1 = '/acw'
    elif sensor_str == 'dc':
        sens_str1 = '/dc_0.05_0.05'
    elif sensor_str == 'pm':
        sens_str1 = '/pm_1.0_1.0'
    else:
        print('Invalid Sensor String for "create_mex_str" function')
        return
    
    # patient_no 22 only does four once
    if patient_no == 22:
        patient_no = 21
    
    if exercise_no == 8:
        str_2 = '/{:0>2d}/04_'.format(patient_no) + sensor_str + '_2.csv'
    else:
        str_2 = '/{:0>2d}/{:0>2d}_'.format(patient_no, exercise_no) + sensor_str + '_1.csv'
    
    location_str = str_1 + sens_str1 + str_2
    
    return location_str


# Downsampling function to reduce amount of accelerometer data.
def down_sample(one_d_array, factor):
    # Get initial array
    ds_array0 = one_d_array[0::factor]
    
    # add the following n values to each element
    for i in range(factor - 1):
        # making sure the arrays align
        new_array = one_d_array[i + 1::factor]
        if len(new_array) != len(ds_array0):
            new_array = np.pad(new_array, ((0, 1), (0, 0)), 'edge')
        ds_array0 = np.add(new_array, ds_array0)
    
    # take the average
    return np.true_divide(ds_array0, factor)
