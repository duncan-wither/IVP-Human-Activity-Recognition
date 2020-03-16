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
from MEX_utils import *

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


def dcnn(train_pair_list, test_pair_list, sens_str='act', pre_str='', samps_per_set=8):
    # constants
    
    # How many times to sample the dataset
    SAMPS_PER_SET = samps_per_set
    
    # Setting the final dimention of the data
    if sens_str == 'dc':
        DATA_DIM_1 = 12
        DATA_DIM_2 = 16
        # picking this many samples (min is 158)
        SET_LENGTH = 150
    elif sens_str == 'pm':
        DATA_DIM_1 = 32
        DATA_DIM_2 = 16
        # picking this many samples (min is 252)
        SET_LENGTH = 240
    else:  # i.e accelerometers
        DATA_DIM_1 = 3
        DATA_DIM_2 = 1
        # picking this many samples (as the smallest dataset is 1739 long)
        SET_LENGTH = 1600
    
    # Randomising the Data
    rd.shuffle(train_pair_list)
    
    # Initialising the matricies
    # training
    data_act_train = np.zeros((len(train_pair_list) * SAMPS_PER_SET, SET_LENGTH, DATA_DIM_1, DATA_DIM_2))
    label_act_train = np.zeros((len(train_pair_list) * SAMPS_PER_SET))
    
    # testing
    data_act_test = np.zeros((len(test_pair_list), SET_LENGTH, DATA_DIM_1, DATA_DIM_2))
    label_act_test = np.zeros((len(test_pair_list)))
    
    # TODO - Look at train-test split using the line @5:13 on this vid: https://www.youtube.com/watch?v=n2MxgXtSMBw
    # Assigning training
    i = 0
    for pair in train_pair_list:
        
        data_str = create_mex_str(pair[0], pair[1], sens_str, pre_str)
        # Reading Raw data from the CSV to temp variable in numpy array form
        dataset_holder = pd.read_csv(data_str).iloc[:, 1:].to_numpy(dtype=float)
        
        # for each in the set super sampling
        for set_sup_samp in range(SAMPS_PER_SET):
            # Geting a start point within the dataset at least SET_LENGTH from the end
            start = rd.randint(0, dataset_holder.shape[0] - (SET_LENGTH + 1))  # +1 for safety
            # Writing to the training array
            
            data_act_train[i, :, :] = dataset_holder[start:(start + SET_LENGTH), :].reshape(SET_LENGTH, DATA_DIM_1,
                                                                                            DATA_DIM_2)
            label_act_train[i] = pair[1]  # get labels
            i += 1
    
    # Making sure that it is a list of testing qs
    if not (isinstance(test_pair_list, list)):
        # print('Please make testing pair list a list, even if it is only one pair long (i.e. use "[]")')
        test_pair_list = [test_pair_list]
    
    # Randomising Order
    rd.shuffle(test_pair_list)
    
    # Assigning Testing
    i = 0
    for pair in test_pair_list:
        data_str = create_mex_str(pair[0], pair[1], sens_str, pre_str)
        # Reading Raw data from the CSV to temp variable in numpy array form
        dataset_holder = pd.read_csv(data_str).iloc[:, 1:].to_numpy(dtype=float)
        
        # Getting Randomised start point
        start = rd.randint(0, dataset_holder.shape[0] - (SET_LENGTH + 1))  # +1 for safety
        
        # Writing to the testing arrays.
        data_act_test[i, :, :] = dataset_holder[start:(start + SET_LENGTH), :].reshape(SET_LENGTH, DATA_DIM_1,
                                                                                       DATA_DIM_2)
        label_act_test[i] = pair[1]  # get labels
        i += 1
    
    # converting the labels to one hot encoding (if required)
    label_train_oh = tf.keras.utils.to_categorical(label_act_train, num_classes=int(label_act_train.max() + 1),
                                                   dtype='float64')[:, 1:]
    label_test_oh = tf.keras.utils.to_categorical(label_act_test, num_classes=int(label_act_train.max() + 1),
                                                  dtype='float64')[:, 1:]
    # print(label_train_oh)
    # print(label_test_oh)
    # return
    
    batch_size = 32
    nb_classes = 7
    nb_epoch = 20
    nb_filters = 32
    nb_pool = 2
    nb_conv = 3

    nb_epoch = 30
    batch_size = 5
    
    model = models.Sequential()
    
    # 2D models
    # if DATA_DIM_2!=1:
    # model.add(layers.Conv2D(nb_filters, nb_conv, nb_conv, border_mode='same', input_shape=(SET_LENGTH, DATA_DIM_1,
    # DATA_DIM_2)))
    if sens_str == 'dc':
        model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding="same",
                                input_shape=(SET_LENGTH, DATA_DIM_1, DATA_DIM_2)))
        model.add(layers.Activation('relu'))
        model.add(layers.Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.75))
        model.add(layers.Flatten())
        model.add(layers.Dense(64))
        model.add(layers.Dropout(0.75))
        model.add(layers.Dense(nb_classes))
        model.add(layers.Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    elif sens_str == 'pm':
        nb_pool = 2
        batch_size = 10
        model.add(layers.Conv2D(filters=96, kernel_size=5, activation='relu', padding="same",
                                input_shape=(SET_LENGTH, DATA_DIM_1, DATA_DIM_2)))
        model.add(layers.Activation('relu'))
        model.add(layers.Convolution2D(filters = nb_filters, kernel_size=5))
        model.add(layers.Activation('relu'))
        model.add(layers.Convolution2D(filters=nb_filters, kernel_size=3))
        model.add(layers.Activation('relu'))
        model.add(layers.Convolution2D(filters = nb_filters, kernel_size=3))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(layers.Activation('relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64))
        model.add(layers.Dropout(0.75))
        
    else:  #Accel Data
        print('whoops')
        
    model.add(layers.Dense(nb_classes))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    model.summary()
    
    
    model.fit(data_act_train, label_train_oh, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
              validation_data=(data_act_test, label_test_oh))
    
    test_loss, test_acc = model.evaluate(data_act_test, label_test_oh, verbose=2)
    return test_acc


'''
# model.compile(optimizer='adam',
#             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics=['accuracy'])

# Creating the CNN model
model = models.Sequential()
# TODO - Look and work out what these numbers mean, and the correct assignment.
# potential guide - https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series
# -classification/
# TODO - Decide on the layers, including looking at the dropout layer

model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding="same",
                        input_shape=(SET_LENGTH, DATA_DIM_1, DATA_DIM_2)))
# that means 32 filters of size 3x3

model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding="same",
                        input_shape=(SET_LENGTH, DATA_DIM_1, DATA_DIM_2)))
# that means 32 filters of size 3x3

model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(8))

model.summary()
print()
print('l_train', label_act_train.shape)
print('l_test', label_act_test.shape)
print('d_train', data_act_train.shape)
print('d_test', data_act_test.shape)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(data_act_train, label_act_train, epochs=40,
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
