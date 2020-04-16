# Multi-modal Human Activity Recognition
Human Activity Recognition (HAR) using the [Multi-Modal Data-set](https://ieee-dataport.org/open-access/mex-multi-modal-exercise-dataset)

Looks at comparing the results between a deep learning classification and kNN with DTW, as part of an assignment for 
the EE581 - Image and Video Processing Class at Strathclyde University.

Mario Emilio Manca and Duncan Wither

## Scripts
 - `Eval_kNN.py` script to evaluate the kNN with DTW on the dataset using several runs of 25:5 partitions of training 
    to testing. This generates the `.png` files which show the effectiveness of the module. The `Eval_Matrix.pckl` file is the
    pre-tested grouping to provide easier dissection of the results.
 - `MM_NN.py` multi input algorithm that takes the data from all four sensors

## Modules
 - `dtw/` contains the dynamic time warping functionality.
 - `knn/` contains the k-nearest neighbor functionality (using a DTW search).
 - `MEX_utils/` contains two functions to make dealing with the MEX dataset easier.
 - `MEX_Visualisation/` contains all the functionalty required to visualise the MEx dataset.
 - `Single_Sensor_DL/` contains all the modules related for the single sensor DL methods.
 - `MM_NN/` contains the multi-input NN functionality, including archetecture.
 - `tests/` contains all the testing files used in developting the functions.
 
## Other Files
 - `Accuracy_vs_K.png` shows the kNN accuracy of the individual sensors using different k values.
 - `Accuracy_vs_K_and_Final.png` same as above but also shows the accuracy of the combined prediction along with the baseline random guess accuracy.
 - `Boxplot.png` boxplot of the kNN confidences.
 - `MM_pre-trained_model.h5` pre-trained multimodal neural network classifier.
 - `Report.pdf` is the report of the project and the outcomes.
 - `requirements.txt` list of the packages reequired to run everthing.
 
### Notes
 - This is designed to work with the MEx dataset placed in a falder titled 'dataset' in the top level of this directory.
 - If using Linux, use tensorflow version 1.15.2 as versions 2.0.0 and 2.1.0 were found to have [memory issues](https://github.com/tensorflow/tensorflow/issues/35030).


