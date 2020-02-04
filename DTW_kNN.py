#!/usr/bin/env python
'''
DTW_kNN.py
Created by Duncan at 02/02/2020

Description: Implenting a kNN algorithm using the DTW as the distance function.
'''
## LIBS
# Default Libs
import pickle
# 3rd Party Libs
import numpy as np
import pandas as pd
import cv2 #opencv-python
# Pathfinding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
# Custom Libs


'''
TODO
1 - Create DTW function w/ two potential datasets
    1a - create map of the euclideian distances at each time between the two images
        1ai - Use the sum of the Euclian distances between the pixels
    1b - find shortest path from the start of both to the end of both
        1bi - no reverse time in either.
        1bii- could be ways to speed this up
    1c - sum? the distances of the paths. Use this as the Distance Algorithm
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
p1ex1 = pd.read_csv('./MEx Dataset/Dataset/dc_0.05_0.05/01/01_dc_1.csv').iloc[:, 1:].to_numpy(dtype=float)
p2ex1 = pd.read_csv('./MEx Dataset/Dataset/dc_0.05_0.05/02/01_dc_1.csv').iloc[:, 1:].to_numpy(dtype=float)
# converted to np array as DataFrames appeared to be slow and cumbersome to use, not to mention superfluous.

calc_MAP = False
# creating an array for the map
# rows is p1, cols is p2
if calc_MAP:
    MAP = np.full([len(p1ex1), len(p2ex1)], 1e200)
    i = 0
    
    for Mrow in range(len(p1ex1)):
        for Mcol in range(len(p2ex1)):
            img1 = p1ex1[Mrow,:]
            img2 = p2ex1[Mcol,:]
            MAP[Mrow,Mcol] = (np.sqrt(abs(np.square(img2) - np.square(img1)))).sum()
            i += 1
        print(100 * i / (len(p1ex1) * len(p2ex1)))
        
    #Saving MAP
    f = open('MAP.pckl', 'wb')
    pickle.dump(MAP, f)
    f.close()
    
else:
    f = open('MAP.pckl', 'rb')
    MAP = pickle.load(f)
    f.close()

#converting map to image
MAP_img = 255*(MAP/MAP.max())
MAP_img[len(p1ex1)-1,0] = 1 # Start Point
MAP_img[0,len(p2ex1)-1] = 1 # End Point
cv2.imwrite('DTW.png',MAP_img)

# Pathfinding
find_path = False
matrix = MAP
grid = Grid(matrix=matrix)

# node in (y,x) format
start = grid.node(0,len(p1ex1)-1)
end = grid.node(len(p2ex1)-1,0)

if find_path:
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)
    
    print('operations:', runs, 'path length:', len(path))
    print(grid.grid_str(path=path, start=start, end=end))
    
    f = open('PATH.pckl', 'wb')
    pickle.dump(path, f)
    f.close()

else:
    f = open('PATH.pckl', 'rb')
    path = pickle.load(f)
    f.close()
    
#Adding the path to the image
for i in range(len(path)):
    MAP_img[path[i][1],path[i][0]] = 0
    
cv2.imwrite('DTW_path.png',MAP_img)