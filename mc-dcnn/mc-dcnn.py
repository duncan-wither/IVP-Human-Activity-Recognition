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

# Reading Data
# Initialising the matricies
#training
train_list = [1,2,3,4,5,6]
data_act_train = np.zeros((len(train_list)*7,1600,3))
label_act_train = np.zeros(len(train_list)*7)
#testing
test_list = [7,8,9,10]
data_act_test = np.zeros((len(test_list)*7,1600,3))
label_act_test = np.zeros(len(test_list)*7)

# picking a random 1600 digits (as the smallest dataset is 1739 long)
ACT_DIGITS = 1600

# Assigning trainign
# './../MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(patient,exercise)
for i in range(len(train_list)):
    for j in range(7):
        data_act_train_str = '../MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(train_list[i],j+1)
        # Reading Raw data from the CSV to temp variable
        dataset_holder = pd.read_csv(data_act_train_str).iloc[:, 1:].to_numpy(dtype=float)
        # Geting a start point within the dataset at least ACT_DIGITS from the end
        start = rd.randint(0,dataset_holder.shape[0] - (ACT_DIGITS+1)) #+1 for safety
        # Writing to the training array
        data_act_train[(7*i+j),:,:] = dataset_holder[start:(start+ACT_DIGITS),:]
        label_act_train[7*i+j] = j #get labels

#Assigning Testing
for i in range(len(test_list)):
    for j in range(7):
        data_act_test_str = '../MEx Dataset/Dataset/act/{:0>2d}/{:0>2d}_act_1.csv'.format(test_list[i],j+1)
        # Reading Raw data from the CSV to temp variable
        dataset_holder = pd.read_csv(data_act_test_str).iloc[:, 1:].to_numpy(dtype=float)
        # Getting Randomised start point
        start = rd.randint(0,dataset_holder.shape[0] - (ACT_DIGITS+1)) #+1 for safety
        # Writing to the testing array.
        data_act_test[7*i+j,:,:] = dataset_holder[start:(start+ACT_DIGITS),:]
        label_act_test[7*i+j] = j #get labels


# Creating the CNN model
model = models.Sequential()

#TODO - Look and work out what these numbers mean, and the correct assignment.
#potential guide - https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
#TODO - Decide on the layers, including looking at the dropout layer

model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(ACT_DIGITS,3)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(32, 3, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(32, 3, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(32, 3, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=2))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
exit()
# DOESNT WORK BEYOND THIS POINT
history = model.fit(data_act_train, label_act_train, epochs=10,
                    validation_data=(data_act_test, label_act_test))

