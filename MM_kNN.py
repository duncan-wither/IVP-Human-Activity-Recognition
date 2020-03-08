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

list_acw = knn.find_costs([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)], (4, 2), verbose=0,
                          down_sample_rate=10, ex_str='acw')
print('List 1 created')
list_act = knn.find_costs([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)], (4, 2), verbose=0,
                          down_sample_rate=10, ex_str='act')
print('List 2 created')
list_dc = knn.find_costs([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)], (4, 2), verbose=0,
                         down_sample_rate=1, ex_str='dc')
print('List 3 created')
list_pm = knn.find_costs([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)], (4, 2), verbose=0,
                         down_sample_rate=1, ex_str='pm')
print('List 4 created')

# saving the results
f = open('mm_knn.pckl', 'wb')
pickle.dump([list_acw, list_act, list_dc, list_pm], f)
f.close()

# THIS IS WHERE THE RESULTS OFF THIS NEED COMBINED IN SOME FORM!
# Initial attempts are shown below,
# Also the simple mode that'd be used is implemented in the kNN module.

k = 5
# for each array take the k top results, concatenate, and then find the mode.
# concatenating the k top results from each array.
results_array = np.vstack((list_acw[:k, :], list_act[:k, :], list_dc[:k, :], list_pm[:k, :]))
# finding the mode.
nearest_neighbor = int(stats.mode(results_array[:, 0])[0][0])
# nearest_neighbor = stats.mode(results_array[:,0])[0][0]

print('Thus it is most similar to exercise {}'.format(nearest_neighbor))
