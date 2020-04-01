# !/usr/bin/env python
"""
DTW.py
Created by Duncan at 02/02/2020

Description: Implementing a kNN algorithm using the DTW as the distance function.
"""
# LIBS
## Default Libs
import math

## 3rd Party Libs
import numpy as np
## Path-finding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

__author__ = "Duncan Wither"
__copyright__ = "Copyright 2020, Duncan Wither"
__credits__ = ["Duncan Wither"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Duncan Wither"
__email__ = ""
__status__ = "Prototype"


# Creating map for the 2D arrays (images)
def create_map(vid_1, vid_2):
    # Create array, initialised with high cost spots (in case any are skipped)
    map_array = np.full([len(vid_1), len(vid_2)], 1e200)
    
    # For each spot find the "cost" of that point.
    r_max, c_max = map_array.shape
    for r_val in range(r_max):
        for c_val in range(c_max):
            img_1 = vid_1[r_val, :]
            img_2 = vid_2[c_val, :]
            # Cost is defined by the sum of the distances of the squares.
            map_array[r_val, c_val] = (np.sqrt(abs(np.square(img_2) - np.square(img_1)))).sum() + 1.0
            # the plus 0.001 is to prevent 0 values which are seen as impassible by the path-finding function.
    return map_array


def create_quick_map(vid_1, vid_2, path_width_percentage, other_vals=1e200):
    # Create array, initialised with high cost spots (in case any are skipped)
    # path_width_percentage is the percentage width of the path
    # other_vals is the value to put in the untouched array values
    # converting other vals to a float
    other_vals = other_vals * 1.0
    
    map_array = np.full([len(vid_1), len(vid_2)], other_vals)
    
    # For each spot find the "cost" of that point.
    r_max, c_max = map_array.shape
    
    # path width percentage is in float [0:1].
    dc = path_width_percentage * c_max  # horisontal path width variation
    
    # constant for the width-height ratio
    k = r_max / c_max
    
    for r_val in range(r_max):
        for c_val in range(math.floor(r_val / k - dc), math.ceil(r_val / k + dc)):
            if c_max > c_val >= 0:
                img_1 = vid_1[r_val, :]
                img_2 = vid_2[c_val, :]
                # Cost is defined by the PSNR non dB measure
                # MSE = (abs(np.square(img_2) - np.square(img_1))).sum()
                # PSNR = MAX^2/MSE
                # Higher PSNR is better, but we want lower, so we'll use 1/PSNR
                # 1/PSNR = MSE/(MAX^2)
                # Given max is the same for all, 1/PSNR is proportional to MSE thus just use MSE:
                map_array[r_val, c_val] = (abs(np.square(img_2) - np.square(img_1))).sum() + 1.0
                # the +1.0 is to prevent 0 values which are seen as impassible by the path-finding function.
    
    # making sure start and end are accessible
    # map_array[r_max-1, c_max-1]=1
    # map_array[0, 0] = 1
    return map_array


def dtw_path(map_array):
    # Create grid using the map array
    grid = Grid(matrix=map_array)
    x_max, y_max = map_array.shape
    
    # Create grid points at start of each, and the end of each
    # node in (y,x) format
    start = grid.node(0, 0)
    end = grid.node(y_max - 1, x_max - 1)
    
    # Path find
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path_dtw, runs = finder.find_path(start, end, grid)
    
    # Return the path
    return path_dtw


def dtw_cost(dtw_map, path_list):
    # convert the path into a cost (which is the sum of the points on the path)
    
    # this value accounts for the changing sizes of the different maps
    length_corr = max(dtw_map.shape)
    
    dtw_cost_val = 0
    for i in range(len(path_list)):
        dtw_cost_val += dtw_map[path_list[i][1], path_list[i][0]]
    return dtw_cost_val / length_corr
