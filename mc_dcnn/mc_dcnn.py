# !/usr/bin/env python
""" mc-dnn.py
Created by slam at 17/02/2020

Description:
"""

import act_CNN
import acw_CNN
import DC_CNN
import PM_CNN
from MM_NN import datasets
from MM_NN import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import locale
import os

inputPath = "../MEx Dataset/Dataset/"
# ============ Loading data =================
print("[INFO] loading thigh accelerometer attributes...")
act_attributes, act_labels = act_CNN.load_ac_attributes_labels(inputPath + 'act', verbose=False)
print("[INFO] loading wrist accelerometer attributes...")
acw_attributes, acw_labels = acw_CNN.load_ac_attributes_labels(inputPath + 'acw', verbose=False)
print("[INFO] loading depth camera images...")
DC_images, DC_labels = DC_CNN.load_DC_images(inputPath + 'dc_0.05_0.05', verbose=False)
print("[INFO] loading pressure mat images...")
PM_images, PM_labels = PM_CNN.load_PM_images(inputPath + 'pm_1.0_1.0', verbose=False)

# ============ Partition data 70 % training, 30 % testing =========================
print("[INFO] constructing training/testing split...")
seed=55
(act_train_attributes, act_test_attributes) = train_test_split(act_attributes, test_size=0.30, random_state=seed)
(act_train_labels, act_test_labels) = train_test_split(act_labels, test_size=0.30, random_state=seed)
(acw_train_attributes, acw_test_attributes) = train_test_split(acw_attributes, test_size=0.30, random_state=seed)
(acw_train_labels, acw_test_labels) = train_test_split(acw_labels, test_size=0.30, random_state=seed)
(DC_train_images, DC_test_images) = train_test_split(DC_images, test_size=0.30, random_state=seed)
(DC_train_labels, DC_test_labels) = train_test_split(DC_labels, test_size=0.30, random_state=seed)
(PM_train_images, PM_test_images) = train_test_split(PM_images, test_size=0.30, random_state=seed)
(PM_train_labels, PM_test_labels) = train_test_split(PM_labels, test_size=0.30, random_state=seed)

# ======== Perform min-max scaling on accelerometer data ===================
print("[INFO] processing data...")
act_trainX_attributes = act_CNN.process_ac_attributes(act_train_attributes)
act_testX_attributes = act_CNN.process_ac_attributes(act_test_attributes)
acw_trainX_attributes = acw_CNN.process_ac_attributes(acw_train_attributes)
acw_testX_attributes = acw_CNN.process_ac_attributes(acw_test_attributes)

# ========= Create CNN models =============================
# MLP for the thigh accelerometer
act_MLP_model = act_CNN.create_cnn()
# MLP for the wrist accelerometer
acw_MLP_model = acw_CNN.create_cnn()
# CNN for the depth camera
DC_CNN_model = DC_CNN.create_cnn(12, 16, 1)
# CNN for the pressure mat
PM_CNN_model = PM_CNN.create_cnn(32, 16, 1)

# ========== Combine models ================
combinedInput = concatenate([act_MLP_model.output, acw_MLP_model.output, DC_CNN_model.output, PM_CNN_model.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
# x = Dense(6, activation="relu")(x)
# x = Dense(8, activation="relu")(x)
# x = Dense(12, activation="relu")(x)
x = Dense(8, activation="softmax")(x)

model = Model(inputs=[act_MLP_model.input, acw_MLP_model.input, DC_CNN_model.input, PM_CNN_model.input], outputs=x)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# # train the model
print("[INFO] training model...")
model.fit([act_trainX_attributes, acw_trainX_attributes, DC_train_images, PM_train_images], act_train_labels, epochs=15,
          verbose=2, validation_data=[[act_testX_attributes, acw_testX_attributes, DC_test_images, PM_test_images],
                                      act_test_labels])

print("[INFO] testing model...")
test_loss, test_acc = model.evaluate([act_testX_attributes, acw_testX_attributes, DC_test_images, PM_test_images],
                                     act_test_labels)

print('\nTest accuracy:', test_acc)


