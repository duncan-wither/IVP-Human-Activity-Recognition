#!/usr/bin/env python
""" test2.py
Description:Testing the dtw function with the accelerometers.
"""
# Libs
import time
import cv2  # opencv-python
import numpy as np
import pandas as pd
from dtw import *


# Functions
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


# Reading the data-set to numpy matrices through pandas
p1ex1 = pd.read_csv('../../MEx Dataset/Dataset/act/01/01_act_1.csv').iloc[:, 1:].to_numpy(dtype=float)
p2ex1 = pd.read_csv('../../MEx Dataset/Dataset/act/02/01_act_1.csv').iloc[:, 1:].to_numpy(dtype=float)

# Need to reduce these down as they are polled every 0.01s, maybe convert to 10ths of a second.
print("Downsapling")
p1ex1 = down_sample(p1ex1, 10)
p2ex1 = down_sample(p2ex1, 10)
print("Done!")

# creating an array for the map
print('Finding Maps')
t0 = time.time()
MAP1 = create_quick_map(p1ex1, p2ex1, 0.2, other_vals=-0.1)
t1 = time.time()
print('Found Map 1 in ', t1 - t0, ' seconds')
t0 = time.time()
MAP2 = create_map(p1ex1, p2ex1)
t1 = time.time()
print('Found Map 2 in ', t1 - t0, ' seconds')

# converting map to image
# having uniform colours between requires same scale factor
img_min = min(MAP1.min(), MAP2.min())
img_max = max(MAP1.max(), MAP2.max())
MAP1_img = 255 * ((MAP1 - img_min) / (img_max - img_min))
MAP2_img = 255 * ((MAP2 - img_min) / (img_max - img_min))

# Pathfinding
print('Finding Paths')
t0 = time.time()
path1 = dtw_path(10 * MAP1)  # 100x as the pathfinder needs > 1
t1 = time.time()
print('Found Path 1 in ', t1 - t0, ' seconds')
t0 = time.time()
path2 = dtw_path(1000 * MAP2)  # 100x as the pathfinder needs > 1
t1 = time.time()
print('Found Path 2 in ', t1 - t0, ' seconds')

# Adding the path to the image
for i in range(len(path1)):
    MAP1_img[path1[i][1], path1[i][0]] = 0
cv2.imwrite('QuickMAP_on.png', MAP1_img)

for i in range(len(path2)):
    MAP2_img[path2[i][1], path2[i][0]] = 0
cv2.imwrite('QuickMAP_off.png', MAP2_img)

# calculating the DTW path costs
DTW_cost_1 = dtw_cost(MAP1, path1)
print('DTW cost 1 = ', DTW_cost_1)

DTW_cost_2 = dtw_cost(MAP2, path2)
print('DTW cost 2 = ', DTW_cost_2)
