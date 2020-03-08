#!/usr/bin/env python
""" test_knn.py
Created by slam at 07/02/2020

Description: Simple test of knn process
"""
# LIBS
## Default Libs
import pickle
import sys

## 3rd Party Libs
import numpy as np
import pandas as pd
from scipy import stats

## Custom Libs
sys.path.append('../../')  # enables seeing the libs above
from dtw import dtw

__author__ = "Duncan Wither"
__copyright__ = "Copyright 2020, Duncan Wither"
__credits__ = ["Duncan Wither"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Duncan Wither"
__email__ = ""
__status__ = "Prototype"


# Function
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


k_val = 5  # (number of nearest neighbors to look at)

training_set = []
# Training set is a list of pairs. Each item is a different dataset. The first value of the pair is the dataset
# array, the second is the type.

# Create Training Set
ex_no = 4  # no of exercises to look at
pat_no = 5  # no of patients to look at
ds_factor = 50  # down sampling rate
for ex in range(ex_no):
    for pat in range(pat_no):
        data_set_str = '../../MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(pat + 1, ex + 1)
        data_set = down_sample(pd.read_csv(data_set_str).iloc[:, 1:].to_numpy(dtype=float), ds_factor)
        training_set.append([data_set, ex + 1])

# Creating dataset using new patient and exercise 2
test_set_str = '../../MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(8, 2)
test_set = down_sample(pd.read_csv(test_set_str).iloc[:, 1:].to_numpy(dtype=float), ds_factor)

# creating array to hold the cost values
costs = np.zeros((ex_no * pat_no, 2))  # array to hold the exercise number (col 1) and costs (col 2)

f = open('COSTS_un.pckl', 'wb')  # storing the unsorted costs list
pickle.dump(costs, f)
f.close()

# finding costs
for t_val in range(ex_no * pat_no):
    # Storing Exercise number
    costs[t_val][0] = training_set[t_val][1]
    # Finding some form of cost function
    # costs[t_val][1] = random.random() #example cost function
    # print(training_set[t_val][0])
    # using the DTW for cost
    MAP = dtw.create_quick_map(training_set[t_val][0], test_set, 0.2, other_vals=-0.1)
    PATH = dtw.dtw_path(100 * MAP)
    costs[t_val][1] = dtw.dtw_cost(MAP, PATH)
    # print(costs[t_val][1])
    print(100 * (t_val + 1) / (ex_no * pat_no), '% Done')

# Sorting Costs
costs = costs[costs[:, 1].argsort()]
print(costs)

f = open('COSTS_sort.pckl', 'wb')  # sorting the sorted costs
pickle.dump(costs, f)
f.close()

print('Thus it is most similar to exercise', int(stats.mode(costs[0:5, 0])[0][0]))

'''
from "https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761"
The KNN Algorithm
1. Load the data
2. Initialize K to your chosen number of neighbors
3. For each example in the data
3.1 Calculate the distance between the query example and the current example from the data.
3.2 Add the distance and the index of the example to an ordered collection
4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
5. Pick the first K entries from the sorted collection
6. Get the labels of the selected K entries
7. If regression, return the mean of the K labels
8. If classification, return the mode of the K labels
'''
