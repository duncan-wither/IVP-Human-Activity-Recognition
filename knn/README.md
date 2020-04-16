# kNN Function
Implemented kNN function as a module. The main reference material is an article from [towards data science](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761).

## Functions
 - `mex_knn` the main function (high level) function running the algorithm.
 - `find_costs` This is the majority of the kNN module. It returns the sorted cost list, allowing for more 
 detailed inspection of the results, and thus more flexibility to combine for multi-modal data.
 - `pick_nn` simple function to pick the k'th nearest neighbour from the sorted costs list. 
 - `resample` creates an array of samples from the original long sample for further use.
 