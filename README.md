# Multi-modal Human Activity Recognition
Human Activity Recognition (HAR) using the [Multi-Modal Data-set](https://ieee-dataport.org/open-access/mex-multi-modal-exercise-dataset)

Looks at comparing the results between a deep learning method (MC-DNN?) and kNN with DTW, as part of an assignment for 
the EE581 - Image and Video Processing Class at Strathclyde University.

Mario Emilio Manca and Duncan Wither

## Files
 - `dtw/` contains the dynamic time warping functionality.
 - `knn/` contains the k-nearest neighbor functionality (using a DTW search).
 - `mc-dcnn/` contains the deep learning classifying functionality.
 - `MM_kNN.py` looks at integrating kNN for multi-modal data.
   - This doesnt appear to be working very effectively at the moment.
 - `tests/` contains all the testing files for the functions.

### Key TODOs
**Bold** is currently active / important problems.
 - [ ] **Visualise Data**
 - [ ] Create Deterministic Method
   - [x] Create DTW Function
   - [X] Create kNN Function
   - [ ] Integrate for multi modal data.
     - [ ] **How to use kNN results for multimodal Data?**
     - [X] How to compare the 2D arrays (dc+pm) in the DTW?
 - [ ] Evaluate Deterministic Method
 - [ ] Create DL Method
   - [x] Create initial working CNN feature extractor
   - [ ] **Create working feature extractor**
   - [ ] Create MLP classifier
 - [ ] Evaluate DL Method
 - [ ] Evaluation Script
   - Runs each method X times with random input sets
   - Times each run, to get accuracy.
   - Takes the % accuract and adds it to a list (and saves list)
   - Prints resutls graphs.
   - use [this](https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python) to log console for results dissection. 
 
 ### Other info
 Use tensorflow ver 1.15.2 as versions 2.0.0 and 2.1.0 have [memory issues](https://github.com/tensorflow/tensorflow/issues/35030)
 
