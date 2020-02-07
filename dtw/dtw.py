# !/usr/bin/env python
"""
DTW.py
Created by Duncan at 02/02/2020

Description: Implementing a kNN algorithm using the DTW as the distance function.
"""
# LIBS
## 3rd Party Libs
import numpy as np
## Path-finding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


# Custom Functions
def create_map(vid_1, vid_2):
    map_array = np.full([len(vid_1), len(vid_2)], 1e200)
    r_max, c_max = map_array.shape
    for r_val in range(r_max):
        for c_val in range(c_max):
            img_1 = vid_1[r_val, :]
            img_2 = vid_2[c_val, :]
            map_array[r_val, c_val] = (np.sqrt(abs(np.square(img_2) - np.square(img_1)))).sum()
    
    return map_array


def dtw_path(map_array):
    grid = Grid(matrix=map_array)
    x_max, y_max = map_array.shape
    
    # node in (y,x) format
    start = grid.node(0, x_max - 1)
    end = grid.node(y_max - 1, 0)
    
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path_dtw, runs = finder.find_path(start, end, grid)
    
    return path_dtw


def dtw_cost(array1, array2):
    dtw_map = create_map(array1, array2)
    dtw_route = dtw_path(dtw_map)
    dtw_cost_val = 0
    for i in range(len(path2)):
        dtw_cost_val += dtw_map[dtw_route[i][1], dtw_route[i][0]]
    return dtw_cost_val
