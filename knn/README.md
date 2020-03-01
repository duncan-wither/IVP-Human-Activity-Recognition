# kNN Function
Implemented kNN function as a module. The main reference material is an article from [towards data science](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761).

## Functions
 - `mex_knn` the main function (high level) function running the algorithm.
 - `find_costs` This is the majority of the kNN module. It returns the sorted cost list, allowing for more 
 detailed inspection of the results, and thus more flexibility to combine for multi-modal data.
 - `pick_nn` simple function to pick the k'th nearest neighbor from the sorted costs list. 
 - `down_sample` works to reduce the amount of data, esp as each accelerometer is operating at 1kHz.
 
## Other Files
All files bar `__init__.py` and `knn.py` are for/from testing
 - `test_knn.py` is just the basic testing of the described knn algorithm
 - `test_knn_module.py` is a test of the `knn` module.

### TODOs
- [ ] Additional Functionality
  - [x] Add ability to change which set is being used (ie D.C. or wrist accelerometer)
  - [ ] [optional] Create function to quickly create lists for training sets.
  - [ ] [optional] Implement a simpler euclidean distance cost function?
- [x] Improve performance, or is this done at a lower level (or higher)?
  - this is best done at a higher level according to [this](https://www.oreilly.com/library/view/the-art-of/9780596802424/ch04.html) advice.
- [x] Develop kNN function into module.
 