#!/usr/bin/env python
""" test_mc-dcnn_module.py
Created by slam at 08/03/2020

Description: Script to test the MC-DCNN module
"""

import random as rd
# Default Libs
import sys

# Custom Libs
import mc_dcnn

# 3rd Party Libs

training_list = []
testing_list = []

#sys.stdout = open('console_log.txt', 'w')



patex_list = rd.sample(range(210), int(210 * 0.7))
for pat in range(30):
    for ex in range(8):
        if (ex + 8 * pat) in patex_list:
            training_list.append((pat + 1, ex + 1))
        else:
            testing_list.append((pat + 1, ex + 1))
print('Runs are DC, DC, PM,PM')
print('#^#^#^ Run 1:')
#a=mc_dcnn.dcnn(training_list, testing_list, sens_str='dc', pre_str='../../')
#print('#^#^#^ Run 2:')
b=mc_dcnn.dcnn(training_list, testing_list, sens_str='dc', pre_str='../../')
#print('#^#^#^ Run 3:')
c=mc_dcnn.dcnn(training_list, testing_list, sens_str='pm', pre_str='../../')
#print('#^#^#^ Run 4:')
#d=mc_dcnn.dcnn(training_list, testing_list, sens_str='pm', pre_str='../../')
print('#^#^#^ Run Results:')
#print(a,'\n',b,'\n',c,'\n',d)
'''



'''
