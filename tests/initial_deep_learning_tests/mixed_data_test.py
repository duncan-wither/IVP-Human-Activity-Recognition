#!/usr/bin/env python
""" mixed_data_test.py
Created by slam at 17/03/2020
based on this : https://heartbeat.fritz.ai/building-a-mixed-data-neural-network-in-keras-to-predict-accident
-locations-d51a63b738cf
"""
# Default Libs
from __future__ import absolute_import, division, print_function, unicode_literals

import random as rd

# 3rd Party Libs
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Custom Libs
from MEX_utils import *


# Custom Function based on the CNN function in:
#https://heartbeat.fritz.ai/building-a-mixed-data-neural-network-in-keras-to-predict-accident-locations-d51a63b738cf
def cnn_for_2d_data(width, height, depth, filters=(16, 32, 64), regularizer=None):
    """
        Creates a CNN with the given input dimension and filter numbers.
        """
    # Initialize the input shape and channel dimension, where the number of channels is the last dimension
    inputShape = (height, width, depth)
    chanDim = -1
    
    x = models.Sequential()
    
    # Define the model input
    inputs = layers.Input(shape=inputShape)
    x.add(inputs)
    # Loop over the number of filters
    for (i, f) in enumerate(filters):
        # If this is the first CONV layer then set the input appropriately
        
        # Create loops of CONV => RELU => BN => POOL layers
        x.add(layers.Conv2D(f, (3, 3), padding="same"))
        x.add(layers.Activation("relu"))
        x.add(layers.BatchNormalization(axis=chanDim))
        x.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Final layers - flatten the volume, then Fully-Connected => RELU => BN => DROPOUT
    x.add(layers.Flatten())
    x.add(layers.Dense(16, kernel_regularizer=regularizer))
    x.add(layers.Activation("relu"))
    x.add(layers.BatchNormalization(axis=chanDim))
    x.add(layers.Dropout(0.5))
    
    # Apply another fully-connected layer, this one to match the number of nodes coming out of the MLP
    x.add(layers.Dense(4, kernel_regularizer=regularizer))
    x.add(layers.Activation("relu"))
    
    # Construct the CNN
    
    # Return the CNN
    return x


def cnn_for_1d_data(width, height, filters=(16, 32, 64), regularizer=None):
    """
        Creates a CNN with the given input dimension and filter numbers.
        """
    depth = 1
    # Initialize the input shape and channel dimension, where the number of channels is the last dimension
    inputShape = (height, width, depth)
    chanDim = -1
    
    x = models.Sequential()
    
    # Define the model input
    inputs = layers.Input(shape=inputShape)
    x.add(inputs)
    # Loop over the number of filters
    for (i, f) in enumerate(filters):
        # If this is the first CONV layer then set the input appropriately
        
        # Create loops of CONV => RELU => BN => POOL layers
        x.add(layers.Conv2D(f, (3, 3), padding="same"))
        x.add(layers.Activation("relu"))
        x.add(layers.BatchNormalization(axis=chanDim))
        # x.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Final layers - flatten the volume, then Fully-Connected => RELU => BN => DROPOUT
    x.add(layers.Flatten())
    x.add(layers.Dense(16, kernel_regularizer=regularizer))
    x.add(layers.Activation("relu"))
    x.add(layers.BatchNormalization(axis=chanDim))
    x.add(layers.Dropout(0.5))
    
    # Apply another fully-connected layer, this one to match the number of nodes coming out of the MLP
    x.add(layers.Dense(4, kernel_regularizer=regularizer))
    x.add(layers.Activation("relu"))
    
    # Construct the CNN
    
    # Return the CNN
    return x


# GETTING THE DATA

# splitting the data
training_list = []
testing_list = []
patex_list = rd.sample(range(210), int(210 * 0.7))
for pat in range(30):
    for ex in range(8):
        if (ex + 8 * pat) in patex_list:
            training_list.append((pat + 1, ex + 1))
        else:
            testing_list.append((pat + 1, ex + 1))
rd.shuffle(training_list)
rd.shuffle(testing_list)

# initialising the arrays
SAMPS_PER_SET = 8
dc_data_act_train = np.zeros((len(training_list) * SAMPS_PER_SET, 150, 12, 16))
dc_label_act_train = np.zeros((len(training_list) * SAMPS_PER_SET))
act_data_act_train = np.zeros((len(training_list) * SAMPS_PER_SET, 1600, 3, 1))
act_label_act_train = np.zeros((len(training_list) * SAMPS_PER_SET))

# testing
dc_data_act_test = np.zeros((len(testing_list), 150, 12, 16))
dc_label_act_test = np.zeros((len(testing_list)))
act_data_act_test = np.zeros((len(testing_list), 1600, 3, 1))
act_label_act_test = np.zeros((len(testing_list)))

SET_LENGTH1 = 150
SET_LENGTH2 = 1600
DATA_DIM_d1 = 12
DATA_DIM_d2 = 16
DATA_DIM_a1 = 3
DATA_DIM_a2 = 1
# TODO - Look at train-test split using the line @5:13 on this vid: https://www.youtube.com/watch?v=n2MxgXtSMBw
# Assigning training data
i = 0
for pair in training_list:
    
    dc_data_str = create_mex_str(pair[0], pair[1], 'dc', '../../')
    act_data_str = create_mex_str(pair[0], pair[1], 'act', '../../')
    # Reading Raw data from the CSV to temp variable in numpy array form
    dc_dataset_holder = pd.read_csv(dc_data_str).iloc[:, 1:].to_numpy(dtype=float)
    act_dataset_holder = pd.read_csv(act_data_str).iloc[:, 1:].to_numpy(dtype=float)
    
    # for each in the set super sampling
    for set_sup_samp in range(SAMPS_PER_SET):
        # Geting a start point within the dataset at least SET_LENGTH from the end
        start_dc = rd.randint(0, dc_dataset_holder.shape[0] - (SET_LENGTH1 + 1))  # +1 for safety
        start_act = rd.randint(0, act_dataset_holder.shape[0] - (SET_LENGTH2 + 1))  # +1 for safety
        # Writing to the training array
        
        dc_data_act_train[i, :, :] = dc_dataset_holder[start_dc:(start_dc + SET_LENGTH1), :].reshape(SET_LENGTH1,
                                                                                                     DATA_DIM_d1,
                                                                                                     DATA_DIM_d2)
        act_data_act_train[i, :, :] = act_dataset_holder[start_act:(start_act + SET_LENGTH2), :].reshape(SET_LENGTH2,
                                                                                                         DATA_DIM_a1,
                                                                                                         DATA_DIM_a2)
        
        dc_label_act_train[i] = pair[1]  # get labels
        act_label_act_train[i] = pair[1]  # get labels
        i += 1

# Assigning Testing
i = 0
for pair in testing_list:
    dc_data_str = create_mex_str(pair[0], pair[1], 'dc', '../../')
    act_data_str = create_mex_str(pair[0], pair[1], 'act', '../../')
    # Reading Raw data from the CSV to temp variable in numpy array form
    dc_dataset_holder = pd.read_csv(dc_data_str).iloc[:, 1:].to_numpy(dtype=float)
    act_dataset_holder = pd.read_csv(act_data_str).iloc[:, 1:].to_numpy(dtype=float)
    
    # Getting Randomised start point
    start_dc = rd.randint(0, dc_dataset_holder.shape[0] - (SET_LENGTH1 + 1))  # +1 for safety
    start_act = rd.randint(0, act_dataset_holder.shape[0] - (SET_LENGTH2 + 1))  # +1 for safety
    
    # Writing to the testing arrays.
    dc_data_act_test[i, :, :] = dc_dataset_holder[start_dc:(start_dc + SET_LENGTH1), :].reshape(SET_LENGTH1,
                                                                                                DATA_DIM_d1,
                                                                                                DATA_DIM_d2)
    act_data_act_test[i, :, :] = act_dataset_holder[start_act:(start_act + SET_LENGTH2), :].reshape(SET_LENGTH2,
                                                                                                    DATA_DIM_a1,
                                                                                                    DATA_DIM_a2)
    
    dc_label_act_test[i] = pair[1]  # get labels
    act_label_act_test[i] = pair[1]  # get labels
    i += 1

# converting the labels to one hot encoding (if required)
dc_label_train_oh = tf.keras.utils.to_categorical(dc_label_act_train, num_classes=int(dc_label_act_train.max() + 1),
                                                  dtype='float64')[:, 1:]
act_label_train_oh = tf.keras.utils.to_categorical(act_label_act_train, num_classes=int(act_label_act_train.max() + 1),
                                                   dtype='float64')[:, 1:]
dc_label_test_oh = tf.keras.utils.to_categorical(dc_label_act_test, num_classes=int(dc_label_act_train.max() + 1),
                                                 dtype='float64')[:, 1:]
act_label_test_oh = tf.keras.utils.to_categorical(act_label_act_test, num_classes=int(act_label_act_train.max() + 1),
                                                  dtype='float64')[:, 1:]
print('GOT DATA')




# DOING THE DEEP LEARNING

# Create the MLP and CNN models
# Had to test the order because it was throwing errors.
# cnn2d = cnn_for_2d_data(SET_LENGTH1,DATA_DIM_d1, DATA_DIM_d2)
# cnn2d = cnn_for_2d_data(SET_LENGTH1,DATA_DIM_d2, DATA_DIM_d1)
cnn2d = cnn_for_2d_data(DATA_DIM_d1, SET_LENGTH1, DATA_DIM_d2)
# cnn2d = cnn_for_2d_data(DATA_DIM_d1,DATA_DIM_d2,SET_LENGTH1)
# cnn2d = cnn_for_2d_data(DATA_DIM_d2,SET_LENGTH1, DATA_DIM_d1)
# cnn2d = cnn_for_2d_data(DATA_DIM_d2,DATA_DIM_d1,SET_LENGTH1)

# This one was also adjuseted
cnn1d = cnn_for_1d_data(3, SET_LENGTH2)

# Create the input to the final set of layers as the output of both the MLP and CNN
combinedInput = layers.concatenate([cnn1d.output, cnn2d.output])

# The final fully-connected layer head will have two dense layers (one relu and one sigmoid)
x = layers.Dense(4, activation="relu")(combinedInput)
x = layers.Dense(1, activation="sigmoid")(x)

# The final model accepts numerical data on the MLP input and images on the CNN input, outputting a single value
model1 = models.Model(inputs=[cnn1d.input, cnn2d.input], outputs=x)

# Compile the model
#model1.compile(loss="binary_crossentropy", metrics=['acc'], optimizer='adam')
model1.compile(loss="mean_squared_error", metrics=['acc'], optimizer='adam')

# Train the model
model1_history = model1.fit(
    [act_data_act_train, dc_data_act_train],
    dc_label_act_train,
    validation_data=([act_data_act_test, dc_data_act_test], dc_label_act_test),
    epochs=50,
    batch_size=10)

