# kNN Function
Implemented kNN function as a module. The main reference is from [towards data science](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761).

## Functions
 - `mex_knn` the main function running the algorithm, currently only running with the thigh accelerometer data.
 - `down_sample` works to reduce the amount of data, esp as each accelerometer is operating at 1kHz.
 
## Other Files
All files bar `__init__.py` and `knn.py` are for/from testing
 - `test_knn` is just the basic testing of the described knn algorithm
 - `test_knn_module` is a test of the module.

### TODOs
- [ ] Additional Functionality
  - [ ] Add ability to change which set is being used (ie D.C. or wrist accelerometer)
  - [ ] Create function to quickly create lists for training sets.
  - [ ] Implement a simpler euclidean distance cost function?
- [ ] Improve performance, or is this done at a lower level?
- [x] Develop kNN function into module.
 