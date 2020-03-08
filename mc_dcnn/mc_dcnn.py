#!/usr/bin/env python
""" mc-dnn.py
Created by slam at 17/02/2020

Description:


Info Sources:

CNN stuff
1D conv on tf: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
Conv example on tf: https://www.tensorflow.org/tutorials/images/cnn
CNN tutorial: https://www.youtube.com/watch?v=umGJ30-15_A&t=298s

"""

# Default Libs
from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import random as rd

import numpy as np
# 3rd Party Libs
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Custom Libs

__author__ = "Duncan Wither"
__copyright__ = "Copyright 2020, Duncan Wither"
__credits__ = ["Duncan Wither"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Duncan Wither"
__email__ = ""
__status__ = "Prototype"

'''
How This Works
 - Each Channel Takes a single channel of multivariate data
 - Combine each channel in a multi-layer perceptron to perform classification

'''


def dcnn(train_pat_list, test_pat_list, ex_list=None, sens_str='act', pre_str='', samps_per_set=8):
    # constants
    # picking this many samples (as the smallest dataset is 1739 long)
    SET_LENGTH = 1600
    # How many times to sample the dataset
    SAMPS_PER_SET = samps_per_set
    
    # default exercise list is all the exercises
    if ex_list is None:
        ex_list = [1, 2, 3, 4, 5, 6, 7]
    
    # Setting the final dimention of the data
    if sens_str == 'dc':
        DATA_DIM = 192  # (128*16)
    elif sens_str == 'pm':
        DATA_DIM = 512  # (32*16)
    else:  # i.e accelerometers
        DATA_DIM = 3
    
    # Initialising the matricies
    # training
    data_act_train = np.zeros((len(train_pat_list) * 7 * SAMPS_PER_SET, SET_LENGTH, 3))
    label_act_train = np.zeros(len(train_pat_list) * 7 * SAMPS_PER_SET)
    
    # testing
    data_act_test = np.zeros((len(test_pat_list) * 7, SET_LENGTH, DATA_DIM))
    label_act_test = np.zeros(len(test_pat_list) * 7)
    
    # Assigning trainign
    for i in range(len(train_pat_list)):
        for j in range(7):
            
            if sens_str == 'dc':
                data_act_train_str = pre_str + 'MEx Dataset/Dataset/dc_0.05_0.05/{:0>2d}/{:0>2d}_dc_1.csv'.format(
                    train_pat_list[i], j + 1)
            elif sens_str == 'pm':
                data_act_train_str = pre_str + 'MEx Dataset/Dataset/pm_0.05_0.05/{:0>2d}/{:0>2d}_act_1.csv'.format(
                    train_pat_list[i], j + 1)
            elif sens_str == 'act':
                data_act_train_str = pre_str + 'MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(
                    train_pat_list[i], j + 1)
            elif sens_str == 'acw':
                data_act_train_str = pre_str + 'MEx Dataset/Dataset/acw/{:0>2d}/{:0>2d}_act_1.csv'.format(
                    train_pat_list[i], j + 1)
            
            #data_act_train_str = 'MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(train_pat_list[i], j + 1)
            # Reading Raw data from the CSV to temp variable
            dataset_holder = pd.read_csv(data_act_train_str).iloc[:, 1:].to_numpy(dtype=float)
            
            for k in range(SAMPS_PER_SET):
                # Geting a start point within the dataset at least SET_LENGTH from the end
                start = rd.randint(0, dataset_holder.shape[0] - (SET_LENGTH + 1))  # +1 for safety
                # Writing to the training array
                data_act_train[((7 * i + j) * SAMPS_PER_SET + k), :, :] = dataset_holder[start:(start + SET_LENGTH), :]
                label_act_train[((7 * i + j) * SAMPS_PER_SET + k)] = j  # get labels
    
    # Assigning Testing
    for i in range(len(test_pat_list)):
        for j in range(7):
            
            if sens_str == 'dc':
                data_act_train_str = pre_str + 'MEx Dataset/Dataset/dc_0.05_0.05/{:0>2d}/{:0>2d}_dc_1.csv'.format(
                    test_pat_list[i], j + 1)
            elif sens_str == 'pm':
                data_act_train_str = pre_str + 'MEx Dataset/Dataset/pm_0.05_0.05/{:0>2d}/{:0>2d}_act_1.csv'.format(
                    test_pat_list[i], j + 1)
            elif sens_str == 'act':
                data_act_train_str = pre_str + 'MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(
                    test_pat_list[i], j + 1)
            elif sens_str == 'acw':
                data_act_train_str = pre_str + 'MEx Dataset/Dataset/acw/{:0>2d}/{:0>2d}_act_1.csv'.format(
                    test_pat_list[i], j + 1)
            
            # Reading Raw data from the CSV to temp variable
            dataset_holder = pd.read_csv(data_act_train_str).iloc[:, 1:].to_numpy(dtype=float)
            # Getting Randomised start point
            start = rd.randint(0, dataset_holder.shape[0] - (SET_LENGTH + 1))  # +1 for safety
            # Writing to the testing array.
            data_act_test[7 * i + j, :, :] = dataset_holder[start:(start + SET_LENGTH), :]
            label_act_test[7 * i + j] = j  # get labels

    print('Gathered Data')
    return
'''
# Creating the CNN model
model = models.Sequential()

# TODO - Look and work out what these numbers mean, and the correct assignment.
# potential guide - https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series
# -classification/
# TODO - Decide on the layers, including looking at the dropout layer

model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding="same", input_shape=(SET_LENGTH, 3)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=2))

model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=2))

model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding="same"))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=2))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()
print()
print('l_train', label_act_train.shape)
print('l_test', label_act_test.shape)
print('d_train', data_act_train.shape)
print('d_test', data_act_test.shape)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(data_act_train, label_act_train, epochs=20,
                    validation_data=(data_act_test, label_act_test))

test_loss, test_acc = model.evaluate(data_act_test, label_act_test, verbose=2)
print(test_acc)

'''
'''
How to save and load models:

models.save('CNN_model')
new_model = models.load_model('saved_model/my_model')
'''
# use  model.predict() to predict a value
# make sure the array size is correct https://stackoverflow.com/questions/43017017/keras-model-predict-for-a-single
# -image
