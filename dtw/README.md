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

## Other Files
All files bar `__init__.py` and `dtw.py` are for/from testing
 - `test_dtw_1.py` - Testing function using the MEx depth camera data.
 - `test_dtw_2.py` - Testing function using the MEx accelerometer data.
 - `MAP.pckl` - Pickle of the two dtw_maps generates
 - `PATH.pckl` - Pickle of the two paths created from these paths.
 - *PNG* files - Images of the DTW function path
 - the suffix *_ac* is using accelerometer data, *_dc* is using depth camera data.
### TODOs
- [ ] Improve performance
  - [X] Reduce search space by reducing the max width of search
  - [ ] Add the numerosity reduction mentioned in the paper [optional]
  - [ ] Improve the speed of the path-finding algorithm.
  - [ ] Implement Numba for speedups
  - [ ] Multithreading
- [x] Develop DTW function into module.
 