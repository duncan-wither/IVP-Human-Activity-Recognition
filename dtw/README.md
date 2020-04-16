# DTW Function
Implemented dynamic time warping (DTW) function based on the example of DTW form the paper "Fast Time Series 
Classification Using Numerosity Reduction" by Xi et al.

## Functions
 - `create_quick_map` creates a limited map between the two datasets, improving the speed of creation, and of path
 finding.
 - `create_map` [legacy] same as create_quick_map, but creates the full map, taking more time in the creation and when 
 searching a path.
 - `dtw_path` finds the lowest cost path using the `pathfinding` module.
 - `dtw_cost` calculates the cost of a given path.

 