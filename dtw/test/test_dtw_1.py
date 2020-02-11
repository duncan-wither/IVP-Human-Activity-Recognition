#!/usr/bin/env python
""" test.py
Created by slam at 07/02/2020

Description: Testing the DTW function using the depth camera data.
"""
# LIBS
## Default Libs
import pickle
import sys

## 3rd Party Libs
import numpy as np
import pandas as pd
import cv2  # opencv-python

## Pathfinding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

## Custom Libs
sys.path.append('../') #enables seeing the libs above
from dtw import *


'''
TODO
*1 - Create DTW function w/ two potential datasets
*    1a - create map of the euclideian distances at each time between the two images
*        1ai - Use the sum of the Euclian distances between the pixels
*    1b - find shortest path from the start of both to the end of both
*        1bi - no reverse time in either.
         1bii- could be ways to speed this up
*    1c - sum? the distances of the paths. Use this as the Distance Algorithm
 2 - Create simple kNN algorithm

Biblio:
https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
https://medium.com/@shachiakyaagba_41915/dynamic-time-warping-with-time-series-1f5c05fb8950
https://cran.r-project.org/web/packages/tsfknn/vignettes/tsfknn.html
'''

# Reading Depth camera data
# Var naming is patient#exercise#
# Data is col 1 = time, col 2-193 is the 16x12 camera feed
# So read the data, ignore the first column and convert to numpy array w/ type float
p1ex1 = pd.read_csv('../MEx Dataset/Dataset/dc_0.05_0.05/01/01_dc_1.csv').iloc[:, 1:].to_numpy(dtype=float)
p1ex2 = pd.read_csv('../MEx Dataset/Dataset/dc_0.05_0.05/01/02_dc_1.csv').iloc[:, 1:].to_numpy(dtype=float)
p2ex1 = pd.read_csv('../MEx Dataset/Dataset/dc_0.05_0.05/02/01_dc_1.csv').iloc[:, 1:].to_numpy(dtype=float)
# converted to np array as DataFrames appeared to be slow and cumbersome to use, not to mention superfluous.


calc_MAP = False
# creating an array for the map
# rows is p1, cols is p2
if calc_MAP:
    #MAP1 = create_quick_map(p1ex1, p2ex1, 0.2, other_vals=-0.1)
    MAP1 = create_map(p1ex1, p2ex1)
    print('Found Map 1')
    #MAP2 = create_quick_map(p1ex2, p2ex1, 0.2, other_vals=-0.1)
    MAP2 = create_map(p1ex2, p2ex1)
    print('Found Map 2')
    
    # Saving MAPs
    f = open('test/MAP_dc.pckl', 'wb')
    pickle.dump([MAP1, MAP2], f)
    f.close()

else:
    # Retriveing MAPS
    f = open('test/MAP_dc.pckl', 'rb')
    MAP1, MAP2 = pickle.load(f)
    f.close()

# converting map to image
# having uniform colours between requires same scale factor
img_min = min(MAP1.min(),MAP2.min())
img_max = max(MAP1.max(),MAP2.max())
MAP1_img = 255 * ((MAP1-img_min)/ (img_max-img_min))
MAP2_img = 255 * ((MAP2-img_min)/ (img_max-img_min))
cv2.imwrite('test/DTW1_dc.png', MAP1_img)
cv2.imwrite('test/DTW2_dc.png', MAP2_img)

# Pathfinding

find_path = True

if find_path:
    
    path1 = dtw_path(MAP1)
    print('Found Path 1')
    path2 = dtw_path(MAP2)
    print('Found Path 2')
    
    f = open('test/PATH_dc.pckl', 'wb')
    pickle.dump([path1, path2], f)
    f.close()

else:
    f = open('test/PATH_dc.pckl', 'rb')
    path1, path2 = pickle.load(f)
    f.close()

# Adding the path to the image
for i in range(len(path1)):
    MAP1_img[path1[i][1], path1[i][0]] = 0
cv2.imwrite('test/DTW1_dc_path.png', MAP1_img)

for i in range(len(path2)):
    MAP2_img[path2[i][1], path2[i][0]] = 0
cv2.imwrite('test/DTW2_dc_path.png', MAP2_img)

# calculating the DTW path costs
DTW_cost_1 = dtw_cost(MAP1, path1)
print('DTW cost 1 = ', DTW_cost_1)

DTW_cost_2 = dtw_cost(MAP2, path2)
print('DTW cost 2 = ', DTW_cost_2)

