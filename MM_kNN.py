#!/usr/bin/env python
""" MM_kNN.py
Created by slam at 16/02/2020

Description:Multi-modal kNN
This function is to find the kNN from several data sets, using the mode from an aggregate.
"""

# Default Libs
import pickle
# 3rd Party Libs
import numpy as np
from scipy import stats
# Custom Libs
import knn

__author__ = "Duncan Wither"
__copyright__ = "Copyright 2020, Duncan Wither"
__credits__ = ["Duncan Wither"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Duncan Wither"
__email__ = ""
__status__ = "Prototype"

list_acw = knn.find_costs([(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)],(4,2), down_sample_rate = 10, ex_str='acw')
list_act = knn.find_costs([(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)],(4,2), down_sample_rate = 10, ex_str='act')
list_dc = knn.find_costs([(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)],(4,2), down_sample_rate = 1, ex_str='dc')
list_pm = knn.find_costs([(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)],(4,2), down_sample_rate = 1, ex_str='pm')

#saving the results
f = open('mm_knn.pckl', 'rb')
pickle.dump([list_acw, list_act,list_dc,list_pm], f)
f.close()

k=5
# for each array take the k top results, concatenate, and then find the mode.
# concatenating the k top results from each array.
results_array = np.vstack((list_acw[:k,:],list_act[:k,:],list_dc[:k,:],list_pm[:k,:]))
# finding the mode.
nearest_neighbor = int(stats.mode(results_array[:,0])[0][0])
#nearest_neighbor = stats.mode(results_array[:,0])[0][0]

print('Thus it is most similar to exercise {}'.format(nearest_neighbor))