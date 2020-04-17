# Multi-Modal Neural Network
This module originally aimed to replicate the functionality shown in 
[this paper](http://link.springer.com/10.1007/978-3-319-08010-9_33) by Zheng et. al. and implement it on the MEx dataset. 
It was found in the course of the work however that using multilayer perceptron's worked significantly better for the accelerometer data than the CNNs suggested.
This is the model used here.

## Datasets
This contains the functionality to extract the data from the MEx dataset.
Each module contains the same basic list of functions:
 - `load_ac_attributes_labels(inputPath, acSensor, verbose=True)`
    - Extracts attributes and labels from the accelerometer sensor chosen (either thigh or wrist)
    - Accepts the input path where the accelerometer data is stored, which accelerometer between thigh and wrist, and a command to show on screen the reading status
    - Goes through the 8 exercises for all the 30 patients and stores the data and labels in two arrays
    - Returns data and labels
 - `process_ac_attributes(train, test)`
    - Processes accelerometer data to improve algorithm performance
    - Accepts the training and testing attributes
    - Performs min-max scaling on them
    - Returns the scaled training and testing attributes
 - `load_DC_images(inputPath, verbose=True)`
    - Loads depth camera images and extracts labels
    - Accepts the path where the depth camera images are stored and a command to show on screen the reading status
    - Extracts each 12x16 frame from each exercise and patient
    - Extracts labels
    - Returns images and labels
 - `load_PM_images(inputPath, verbose=True)`
    - Loads pressure mat images and extracts labels
    - Accepts the path where the pressure mat images are stored and a command to show on screen the reading status
    - Extracts each 32x16 frame from each exercise and patient
    - Extracts labels
    - Returns images and labels
    
## Models
This contains the architecture of the MM-NN.
 - `create_act_mlp(dim)` 
    - Accepts the dimension of the thigh accelerometer data
    - Creates an MLP model specifically optimised for the thigh accelerometer
    - Return the model
 - `create_acw_mlp(dim)` 
    - Accepts the dimension of the wrist accelerometer data
    - Creates an MLP model specifically optimised for the wrist accelerometer
    - Return the model
 - `create_DC_cnn(height, width, depth, filters=(16, 32, 64))`
    - Accepts the depth camera data dimensions and the filters
    - Creates a CNN model for the depth camera data
    - Returns the model
- `create_PM_cnn(height, width, depth, filters=(16, 32, 64))`
    - Accepts the pressure mat data dimensions and the filters
    - Creates a CNN model for the pressure mat data
    - Returns the model
    
    
