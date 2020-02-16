# Multi-modal Human Activity Recognition
Human Activity Recognition (HAR) using the [Multi-Modal Data-set](https://ieee-dataport.org/open-access/mex-multi-modal-exercise-dataset)

Looks at comparing the results between a deep learning method (MC-DNN?) and kNN with DTW, as part of an assignment for 
the EE581 - Image and Video Processing Class at Strathclyde University.

Mario Emilio Manca and Duncan Wither

## Files
 - `dtw/` contains the dynamic time warping functionality.
 - `knn/` contains the k-nearest neighbor functionality (using a DTW search).
 - `MM_kNN.py` looks at integrating kNN for multi-modal data.
   - This doesnt appear to be working very effectively at the moment.
 - `tests/` contains all the testing files for the functions.

### TODOs
 - [ ] Visualise Data
 - [ ] Create Deterministic Method
   - [x] Create DTW Function
     - [x] Create DTW Module
     - [ ] Improve Speed
   - [ ] Create kNN Function
     - [X] Create basic kNN function
     - [X] Convert to Module
     - [X] Test Module
     - [ ] Check Speed and improve if needed.
   - [ ] Integrate for multi modal data.
 - [ ] Evaluate Deterministic Method
 - [ ] Create DL Method
 - [ ] Evaluate DL Method
