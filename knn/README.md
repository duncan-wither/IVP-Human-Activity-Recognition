# kNN Function
Implemented kNN function as a module. The main reference material is an article from [towards data science](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761).

## Functions
 - `mex_knn` the main function (high level) function running the algorithm.
 - `find_costs` This is the majority of the kNN module. It returns the sorted cost list, allowing for more 
 detailed inspection of the results, and thus more flexibility to combine for multi-modal data.
 - `pick_nn` simple function to pick the k'th nearest neighbor from the sorted costs list. 
 - `resample` creates an array of samples from the original long sample for further use.
 
### TODOs
- [ ] Additional Functionality
  - [x] Add ability to change which set is being used (ie D.C. or wrist accelerometer)
  - [ ] [optional] Create function to quickly create lists for training sets.
  - [ ] [optional] Implement a simpler euclidean distance cost function?
- [x] Improve performance, or is this done at a lower level (or higher)?
  - this is best done at a higher level according to [this](https://www.oreilly.com/library/view/the-art-of/9780596802424/ch04.html) advice.
- [X] Improve performace by approprietly sampling data
    - [X] split each set into segments of rougly 1.5 seconds.
- [x] Develop kNN function into module.
- [ ] Add input for sampling length
   - [ ] Look at using *args **kwargs framework.
 