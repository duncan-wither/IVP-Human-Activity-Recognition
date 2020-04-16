#!/usr/bin/env python
""" knn.py
Description: k-Nearest Neighbour module.
"""
# Libs
import random as rd
import pandas as pd
from scipy import stats
import dtw
from MEX_utils import *

# Function
# Downsampling function to reduce amount of accelerometer data.

def re_sample(input_data, samples_per_set, no_of_samples, method='first'):
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
        if no_of_samples >= length:
            print("Too many samples for the dataset. Please consider reducing the downsampling rate.")

        if method == 'first':
            start = 0
        elif method == 'rand':
            start = rd.randint(0, length - no_of_samples)
        else:
            print("Incorrect Method input, Using 'First' set")
            start = 0

        # Writing to the output array
        re_sampled_set[i, :, :] = input_data[start:(start + no_of_samples), :]

    return re_sampled_set


def find_costs(pat_ex_pair_list_train, pat_ex_pair_test, down_sample_rate=1, verbose=True, sens_str='act', pre_str='',
               re_sampling='first'):
    # give pairs of patients and exercises, and a pair for the testing
    # take mode from top k values
    # optionally down sample.
    no_of_train_sets = len(pat_ex_pair_list_train)
    training_set = []

    # Setting up SAMP_LEN
    if sens_str == 'dc':
        SAMP_LEN = 10 * 5  # 15Hz with 5s sampling
    elif sens_str == 'pm':
        SAMP_LEN = 10 * 5  # 15Hz with 5s sampling
    else:  # default is acceleromiter data
        SAMP_LEN = 10 * 5  # 10Hz (100Hz 10x down sampled) with 5s sampling

    # Number of samples to take from each set (min is 158 samples, 158/50 samps = 3.16 sets)
    SET_SAMPLES = 1

    # Creating the training set array
    for i in range(no_of_train_sets):
        patient = pat_ex_pair_list_train[i][0]
        exercise = pat_ex_pair_list_train[i][1]

        data_set_str = create_mex_str(patient, exercise, sens_str, pre_str=pre_str)

        if down_sample_rate > 1:
            data_set = down_sample(pd.read_csv(data_set_str).iloc[:, 1:].to_numpy(dtype=float), down_sample_rate)
            # print('Down Sampling')
        else:
            data_set = pd.read_csv(data_set_str).iloc[:, 1:].to_numpy(dtype=float)

        # re-sampling
        data_set_resampled = re_sample(data_set, SET_SAMPLES, SAMP_LEN, re_sampling)
        for j in range(SET_SAMPLES):
            training_set.append([data_set_resampled[j, :, :], exercise * 1.0])

    # Testing String
    test_set_str = create_mex_str(pat_ex_pair_test[0], pat_ex_pair_test[1], sens_str, pre_str=pre_str)

    # Downsampling if Req.
    if down_sample_rate > 1:
        test_set_temp = down_sample(pd.read_csv(test_set_str).iloc[:, 1:].to_numpy(dtype=float), down_sample_rate)
    else:
        test_set_temp = pd.read_csv(test_set_str).iloc[:, 1:].to_numpy(dtype=float)

    # Creating multiple testing sets.
    test_set = []
    test_set_resampled = re_sample(test_set_temp, SET_SAMPLES, SAMP_LEN, re_sampling)
    for j in range(SET_SAMPLES):
        test_set.append(test_set_resampled[j, :, :])

    # len(test_set)

    # creating array to hold the cost values
    costs = np.zeros(
        (len(training_set) * len(test_set), 2))  # array to hold the exercise number (col 1) and costs (col 2)

    i = 0
    for train_set_no in range(len(training_set)):
        for test_set_no in range(len(test_set)):
            # Storing Exercise number
            costs[i][0] = float(training_set[train_set_no][1])
            ## DTW Cost
            MAP = dtw.create_quick_map(training_set[train_set_no][0], test_set[test_set_no], 0.2, other_vals=-0.1)
            PATH = dtw.dtw_path(100 * MAP + 1)
            costs[i][1] = dtw.dtw_cost(MAP, PATH)
            # if costs[i][1] < 0.3:
            #    print( test_set_no, train_set_no)
            # print(training_set[train_set_no][1], costs[train_set_no][1])
            i += 1
            # print(costs[train_set_no][1])
            # print(i)

        if verbose:
            print('Finding Costs is {:6.2f}% Done'.format(100 * (train_set_no + 1) / len(training_set)))

    # Sorting Costs
    costs = costs[costs[:, 1].argsort()]
    # print(costs.shape)
    # print(costs[0:2])
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
