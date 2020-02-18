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

### Key TODOs
**Bold** is currently active / important problems.
 - [ ] **Visualise Data**
 - [ ] Create Deterministic Method
   - [x] Create DTW Function
   - [X] Create kNN Function
   - [ ] Integrate for multi modal data.
     - [ ] **How to use kNN results for multimodal Data?**
     - [ ] **How to compare the 2D arrays (dc+pm) in the DTW?**
 - [ ] Evaluate Deterministic Method
 - [ ] Create DL Method
   - [ ] **Create Working feature Extractor**
   - [ ] Create MLP classifier
 - [ ] Evaluate DL Method
 
 ### Other info
 - Libraries required for the python code:
   - pandas
   - numpy
   - scipy
   - pathfinding
   - tensorflow (ver 1.15.2)
     - versions 2.0.0 and 2.1.0 have [memory issues](https://github.com/tensorflow/tensorflow/issues/35030)
   - matplotlib
