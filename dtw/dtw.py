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

__author__ = "Duncan Wither"
__copyright__ = "Copyright 2020, Duncan Wither"
__credits__ = ["Duncan Wither"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Duncan Wither"
__email__ = ""
__status__ = "Prototype"

# Custom Functions
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
            map_array[r_val, c_val] = (np.sqrt(abs(np.square(img_2) - np.square(img_1)))).sum()+0.001
            # the plus 0.001 is to prevent 0 values which are seen as impassible by the path-finding function.
    return map_array


def dtw_path(map_array):
    # Create grid using the map array
    grid = Grid(matrix=map_array)
    x_max, y_max = map_array.shape
    
    # Create grid points at start of each, and the end of each
    # node in (y,x) format
    start = grid.node(0, x_max - 1)
    end = grid.node(y_max - 1, 0)
    
    # Path find
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path_dtw, runs = finder.find_path(start, end, grid)
    
    # Return the path
    return path_dtw


def dtw_cost(array1, array2):
    #Create the mapping between the arrays
    dtw_map = create_map(array1, array2)
    #Finc the lowest cost path for the route
    dtw_route = dtw_path(dtw_map)
    #convert the path into a cost (which is the sum of the points on the path)
    dtw_cost_val = 0
    for i in range(len(dtw_route)):
        dtw_cost_val += dtw_map[dtw_route[i][1], dtw_route[i][0]]
    return dtw_cost_val
