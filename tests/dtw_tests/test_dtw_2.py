#!/usr/bin/env python
""" test2.py
Description:Testing the dtw function with the accelerometers.
"""
# Libs
import pickle
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
p1ex2 = pd.read_csv('../../MEx Dataset/Dataset/act/01/02_act_1.csv').iloc[:, 1:].to_numpy(dtype=float)
p2ex1 = pd.read_csv('../../MEx Dataset/Dataset/act/02/01_act_1.csv').iloc[:, 1:].to_numpy(dtype=float)

# Need to reduce these down as they are polled every 0.01s, maybe convert to 10ths of a second.
print("Downsapling")
p1ex1 = down_sample(p1ex1, 10)
p1ex2 = down_sample(p1ex2, 10)
p2ex1 = down_sample(p2ex1, 10)
print("Done!")

calc_MAP = True
# creating an array for the map
# rows is p1, cols is p2
if calc_MAP:
    print('Finding Maps')
    # Compating the code functions
    # t_0 = time.time()
    MAP1 = create_quick_map(p1ex1, p2ex1, 0.2, other_vals=-0.1)
    # t_1 = time.time()
    # MAP1 = create_map(p1ex1, p2ex1)
    # t_2 = time.time()
    # print('Quick Map took:',t_1-t_0,'     Normal Map took:',t_2-t_1)
    print('Found Map 1')
    MAP2 = create_quick_map(p1ex2, p2ex1, 0.2, other_vals=-0.1)
    print('Found Map 2')
    # Saving MAPs
    f = open('MAP_ac.pckl', 'wb')
    pickle.dump([MAP1, MAP2], f)
    f.close()

else:
    # Retriveing MAPS
    f = open('MAP_ac.pckl', 'rb')
    MAP1, MAP2 = pickle.load(f)
    f.close()

# converting map to image
# having uniform colours between requires same scale factor
img_min = min(MAP1.min(), MAP2.min())
img_max = max(MAP1.max(), MAP2.max())
MAP1_img = 255 * ((MAP1 - img_min) / (img_max - img_min))
MAP2_img = 255 * ((MAP2 - img_min) / (img_max - img_min))
cv2.imwrite('DTW1_ac.png', MAP1_img)
cv2.imwrite('DTW2_ac.png', MAP2_img)

# Pathfinding

find_path = True

if find_path:
    print('Finding Paths')

    # Timing the Paths
    # t_0 = time.time()
    # path1q = dtw_path(100 * MAP1_q)  # 100x as the pathfinder needs > 1
    # t_1 = time.time()
    path1 = dtw_path(10 * MAP1)  # 100x as the pathfinder needs > 1
    # t_2 = time.time()
    # print('Quick Map Pathing took:',t_1-t_0,'     Normal Map Pathing took:',t_2-t_1)

    print('Found Path 1')
    path2 = dtw_path(1000 * MAP2)  # 100x as the pathfinder needs > 1
    print('Found Path 2')

    # Saving For later
    f = open('PATH_ac.pckl', 'wb')
    pickle.dump([path1, path2], f)
    f.close()

else:
    f = open('PATH_ac.pckl', 'rb')
    path1, path2 = pickle.load(f)
    f.close()

# Adding the path to the image
for i in range(len(path1)):
    MAP1_img[path1[i][1], path1[i][0]] = 0
cv2.imwrite('DTW1_ac_path.png', MAP1_img)

for i in range(len(path2)):
    MAP2_img[path2[i][1], path2[i][0]] = 0
cv2.imwrite('DTW2_ac_path.png', MAP2_img)

# calculating the DTW path costs
DTW_cost_1 = dtw_cost(MAP1, path1)
print('DTW cost 1 = ', DTW_cost_1)

DTW_cost_2 = dtw_cost(MAP2, path2)
print('DTW cost 2 = ', DTW_cost_2)

if DTW_cost_1 < DTW_cost_2:
    print('Sample is closer to first value (exercise 1)')
else:
    print('Sample is closer to second value (exercise 2)')
