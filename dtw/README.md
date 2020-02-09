# DTW Function
Implemented discreet time warping (DTW) function based on the example of DTW form the paper "Fast Time Series 
Classification Using Numerosity Reduction" by Xi et al.

## Functions
dtw_map
dtw_path
dtw_
## Other Files
All files bar __init__.py and dtw.py are for/from testing
 - test.py - Testing function using the MEx depth camera data.
 - test2.py - Testing function using the MEx accelerometer data.
 - MAP.pckl - Pickle of the two dtw_maps generates
 - PATH.pckl - Pickle of the two paths created from these paths.
 - PNG files - Images of the DTW function path
### TODOs
- [ ] Improve performance
  - [ ] Reduce search space by reducing the max width of search
  - [ ] Add the numerosity reduction mentioned in the paper [optional]
  - [ ] Improve the speed of the path-finding algorithm.
  - [ ] Implement Numba for speedups?
- [x] Develop DTW function into module.
 