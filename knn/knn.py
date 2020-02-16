#!/usr/bin/env python
""" knn.py
Created by slam at 07/02/2020

Description: k-Nearest Neighbour module.
"""
# LIBS
## Default Libs
import pickle
import random
import sys

## 3rd Party Libs
import numpy as np
import pandas as pd
from scipy import stats

## Custom Libs
sys.path.append('../../') #enables seeing the libs above
import dtw

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

def find_costs(pat_ex_pair_list_train, pat_ex_pair_test, down_sample_rate = 1,verbose = True, ex_str='act'):
    #give pairs of patients and exercises, and a pair for the testing
    #take mode from top k values
    #optionally down sample.
    
    no_of_train_sets = len(pat_ex_pair_list_train)
    training_set = []
    
    #Setting up which exercise to use
    if ex_str == 'act':
        ex_str1 = 'act'
        ex_str2 = 'act'
    elif ex_str ==  'acw':
        ex_str1 = 'acw'
        ex_str2 = 'acw'
    elif ex_str == 'dc':
        ex_str1 = 'dc_0.05_0.05'
        ex_str2 = 'dc'
    elif ex_str == 'pm':
        ex_str1 = 'pm_1.0_1.0'
        ex_str2 = 'pm'
    else: #default to act
        ex_str1 = 'act'
        ex_str2 = 'act'
    
    base_str_1 = '../../MEx Dataset/Dataset/'+ex_str+'/'
    base_str_2 = '_'+ex_str+'_1.csv'
    
    #Creating the training set array
    for i in range(no_of_train_sets):
        patient = pat_ex_pair_list_train[i][0]
        exercise = pat_ex_pair_list_train[i][1]
        num_str_1 = '{:0>2d}'.format(patient)
        num_str_2 = '{:0>2d}'.format(exercise)
        
        data_set_str = base_str_1 + num_str_1 + '/'+num_str_2+base_str_2
        #'./../MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(patient,exercise)
        
        if down_sample_rate>1:
            data_set = down_sample(pd.read_csv(data_set_str).iloc[:, 1:].to_numpy(dtype=float), down_sample_rate)
            #print('Down Sampling')
        else:
            data_set = pd.read_csv(data_set_str).iloc[:, 1:].to_numpy(dtype=float)
        
        training_set.append([data_set, exercise*1.0])

    test_set_str = '../../MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(pat_ex_pair_test[0], pat_ex_pair_test[1])
    
    if down_sample_rate > 1:
        test_set = down_sample(pd.read_csv(test_set_str).iloc[:, 1:].to_numpy(dtype=float), down_sample_rate)
    else:
        test_set = pd.read_csv(test_set_str).iloc[:, 1:].to_numpy(dtype=float)

    # creating array to hold the cost values
    costs = np.zeros((no_of_train_sets, 2))  # array to hold the exercise number (col 1) and costs (col 2)

    for t_val in range(no_of_train_sets):
        # Storing Exercise number
        costs[t_val][0] = training_set[t_val][1]
        
        # Finding some form of cost function
        ## Random Cost
        # costs[t_val][1] = random.random() #example cost function
        ## DTW Cost
        MAP = dtw.create_quick_map(training_set[t_val][0], test_set, 0.2, other_vals=-0.1)
        PATH = dtw.dtw_path(100 * MAP)
        costs[t_val][1] = dtw.dtw_cost(MAP, PATH)
        
        # print(costs[t_val][1])
        if verbose:
            print('Finding Costs is {:6.2f}% Done'.format(100 * (t_val + 1) / no_of_train_sets))

        # Sorting Costs
        costs = costs[costs[:, 1].argsort()]
        
    return costs
    

def pick_nn(cost_array, k, verbose = True):
    #Getting the mode of the k top matches
    nearest_neighbor = int(stats.mode(cost_array[0:k, 0])[0][0])
    if verbose:
        print('Thus it is most similar to exercise {}'.format(nearest_neighbor))
        
    return nearest_neighbor

def mex_knn(pat_ex_pair_list_train, pat_ex_pair_test, k, down_sample_rate = 1,verbose = True, ex_str='act'):
    costs = find_costs(pat_ex_pair_list_train, pat_ex_pair_test, down_sample_rate,verbose, ex_str)
    nn = sort_costs(costs, k)
    return nn
    
    
    
    
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