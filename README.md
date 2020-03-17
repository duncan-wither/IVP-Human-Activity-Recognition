# Multi-modal Human Activity Recognition
Human Activity Recognition (HAR) using the [Multi-Modal Data-set](https://ieee-dataport.org/open-access/mex-multi-modal-exercise-dataset)

Looks at comparing the results between a deep learning method (MC-DNN?) and kNN with DTW, as part of an assignment for 
the EE581 - Image and Video Processing Class at Strathclyde University.

Mario Emilio Manca and Duncan Wither

## Files
 - `dtw/` contains the dynamic time warping functionality.
 - `knn/` contains the k-nearest neighbor functionality (using a DTW search).
 - `MEX_utils/` contains two functions to make dealing with the MEX dataset easier.
 - `mc-dcnn/` contains the deep learning classifying functionality.
 - `MM_kNN.py` looks at integrating kNN for multi-modal data.
   - This doesnt appear to be working very effectively at the moment.
 - `tests/` contains all the testing files for the functions.
 - `Visual_DC.py` takes the output of the depth camera and makes a time lapse video.
 - `Visual_PM.py` data from the pressure mat is normalised and treated as an image, as for the depth camera the frames are put together in a time lapse video.
 - `Visual_Acc.py` prints on a 3D graph the data from the two accelerometers, based on the user input.   
 - `Eval_kNN.py` script to evaluate the kNN with DTW on the dataset using several runs of 70:30 partitions of training 
    to testing. This generates the `.png` files which show the effectiveness of the module. The `.pckl` file is the
    pre-tested grouping to provide easier dissection of the results.

### Key TODOs
**Bold** is currently active / important problems.
 - [X] Visualise Data
 - [X] Create Deterministic Method
   - [x] Create DTW Function
   - [X] Create kNN Function
   - [X] Integrate for multi modal data.
     - [X] How to use kNN results for multimodal Data
     - [X] How to compare the 2D arrays (dc+pm) in the DTW?
 - [x] Evaluate Deterministic Method
 - [ ] Create DL Method
   - [x] Create initial working CNN feature extractor
   - [ ] Create working feature extractor
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
 
