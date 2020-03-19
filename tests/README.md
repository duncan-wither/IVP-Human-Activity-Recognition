# Test Folder
Contains testing files (and some results).
## DTW Tests
 - `test_dtw_1.py` - Testing function using the MEx depth camera data.
 - `test_dtw_2.py` - Testing function using the MEx accelerometer data.
 - `quick_map_test.py` - Testing the re-written quick-map function.
 - `MAP.pckl` - Pickle of the two dtw_maps generates
 - `PATH.pckl` - Pickle of the two paths created from these paths.
 - *PNG* files - Images of the DTW function path
 - the suffix *_ac* is using accelerometer data, *_dc* is using depth camera data.
## kNN Tests
 - `test_knn.py` is just the basic testing of the described knn algorithm
 - `test_knn_module.py` is a test of the `knn` module.
 - `Test Pickle Files/` is the folder containing the original results files. The tree *.py* scripts were run
 in unison as a form of crude multitasking.
