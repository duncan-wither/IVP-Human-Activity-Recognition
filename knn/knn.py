#!/usr/bin/env python
""" knn.py
Created by slam at 07/02/2020

Description: k-Nearest Neighbour module.
"""
# LIBS
## Default Libs
import random as rd
import sys
## 3rd Party Libs
import numpy as np
import pandas as pd
from scipy import stats

## Custom Libs
# sys.path.append('../../') #(for testing) enables seeing the libs above
import dtw

__author__ = "Duncan Wither"
__copyright__ = "Copyright 2020, Duncan Wither"
__credits__ = ["Duncan Wither"]
__license__ = ""
__version__ = "1.1"
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


# TODO - Create function to make samples from the long datasets
def re_sample(input_data, samples_per_set, no_of_samples):
    # Internal function to re-sample data to reduce individual set size, and re-sample data.
    # works by spliting the input set into a number of samples of set length.
    
    # constants
    length = input_data.shape[0]
    breadth = input_data.shape[1]
    
    # creating output set
    re_sampled_set = np.zeros((samples_per_set, no_of_samples, breadth))
    
    # for each in the no of samples
    for i in range(samples_per_set):
        # Geting a start point within the dataset at least SET_LENGTH from the end
        start = rd.randint(0, length - no_of_samples)
        # Writing to the output array
        re_sampled_set[i, :, :] = input_data[start:(start + no_of_samples), :]
    
    return re_sampled_set


def find_costs(pat_ex_pair_list_train, pat_ex_pair_test, down_sample_rate=1, verbose=True, sens_str='act', pre_str=''):
    # give pairs of patients and exercises, and a pair for the testing
    # take mode from top k values
    # optionally down sample.
    no_of_train_sets = len(pat_ex_pair_list_train)
    training_set = []
    
    # Setting up which exercise to use
    if sens_str == 'act':
        sens_str1 = 'act'
        sens_str2 = 'act'
        SAMP_LEN = 10*5 #10Hz (100Hz 10x down sampled) with 5s sampling
    elif sens_str == 'acw':
        sens_str1 = 'acw'
        sens_str2 = 'acw'
        SAMP_LEN = 15*5 #10Hz (100Hz 10x down sampled) with 5s sampling
    elif sens_str == 'dc':
        sens_str1 = 'dc_0.05_0.05'
        sens_str2 = 'dc'
        SAMP_LEN = 15*5 #15Hz with 5s sampling
    elif sens_str == 'pm':
        sens_str1 = 'pm_1.0_1.0'
        sens_str2 = 'pm'
        SAMP_LEN = 10*5 #15Hz with 5s sampling
    else:  # default to act
        sens_str1 = 'act'
        sens_str2 = 'act'
        SAMP_LEN = 10*5 #10Hz (100Hz 10x down sampled) with 5s sampling
        
    # Number of samples to take from each set (min is 158 samples, 158/50 samps = 3.16 sets, use 5 sets)
    SET_SAMPLES = 5
    
    base_str_1 = pre_str + 'MEx Dataset/Dataset/' + sens_str1 + '/'
    base_str_2 = '_' + sens_str2 + '_1.csv'
    
    # Creating the training set array
    for i in range(no_of_train_sets):
        patient = pat_ex_pair_list_train[i][0]
        exercise = pat_ex_pair_list_train[i][1]
        num_str_1 = '{:0>2d}'.format(patient)
        num_str_2 = '{:0>2d}'.format(exercise)
        
        data_set_str = base_str_1 + num_str_1 + '/' + num_str_2 + base_str_2
        # './../MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(patient,exercise)
        
        if down_sample_rate > 1:
            data_set = down_sample(pd.read_csv(data_set_str).iloc[:, 1:].to_numpy(dtype=float), down_sample_rate)
            # print('Down Sampling')
        else:
            data_set = pd.read_csv(data_set_str).iloc[:, 1:].to_numpy(dtype=float)
        
        data_set_resampled = re_sample(data_set,SET_SAMPLES,SAMP_LEN)
        for j in range(SET_SAMPLES):
            training_set.append([data_set_resampled[j,:,:], exercise * 1.0])
    
    # for testing
    num_str_1 = '{:0>2d}'.format(pat_ex_pair_test[0])
    num_str_2 = '{:0>2d}'.format(pat_ex_pair_test[1])
    test_set_str = base_str_1 + num_str_1 + '/' + num_str_2 + base_str_2
    
    #Downsampling if Req.
    if down_sample_rate > 1:
        test_set_temp = down_sample(pd.read_csv(test_set_str).iloc[:, 1:].to_numpy(dtype=float), down_sample_rate)
    else:
        test_set_temp = pd.read_csv(test_set_str).iloc[:, 1:].to_numpy(dtype=float)
    
    # Creating multiple testing sets.
    test_set = []
    test_set_resampled = re_sample(test_set_temp, SET_SAMPLES, SAMP_LEN)
    for j in range(SET_SAMPLES):
        test_set.append(test_set_resampled[j, :, :])
    
    #len(test_set)
    
    # creating array to hold the cost values
    costs = np.zeros((len(training_set)*len(test_set), 2))  # array to hold the exercise number (col 1) and costs (col 2)
    
    i=0
    for train_set_no in range(len(training_set)):
        for test_set_no in range(len(test_set)):
            # Storing Exercise number
            costs[i][0] = training_set[train_set_no][1]
            ## DTW Cost
            MAP = dtw.create_quick_map(training_set[train_set_no][0], test_set[test_set_no], 0.2, other_vals=-0.1)
            PATH = dtw.dtw_path(100 * MAP + 1)
            costs[train_set_no][1] = dtw.dtw_cost(MAP, PATH)
            i+=1
        # print(costs[train_set_no][1])
        
        if verbose:
            print('Finding Costs is {:6.2f}% Done'.format(100 * (train_set_no + 1) / len(training_set)))
        
    # Sorting Costs
    costs = costs[costs[:, 0].argsort()]
    
    return costs


def pick_nn(cost_array, k, verbose=True):
    # Getting the mode of the k top matches
    nearest_neighbor = int(stats.mode(cost_array[0:k, 0])[0][0])
    if verbose:
        print('Thus it is most similar to exercise {}'.format(nearest_neighbor))
    
    return nearest_neighbor


def mex_knn(pat_ex_pair_list_train, pat_ex_pair_test, k, down_sample_rate=1, verbose=True, sens_str='act', pre_str=''):
    costs = find_costs(pat_ex_pair_list_train, pat_ex_pair_test, down_sample_rate, verbose, sens_str, pre_str)
    nn = pick_nn(costs, k)
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
