# Single Sensor Deep Learning Classifiers

## Scripts
 - `Eval_act_CNN.py` takes the thigh accelerometer data and implements an CNN
 - `Eval_acw_CNN.py` takes the wrist accelerometer data and implements an CNN
 - `Eval_act_MLP.py` takes the thigh accelerometer data and implements an MLP
 - `Eval_acw_MLP.py` takes the wrist accelerometer data and implements an MLP
 - `Eval_DC_CNN.py` takes the depth camera data and implements a CNN
 - `Eval_PM_CNN.py` takes the pressure mat data and implements a CNN
 
## Modules
 - `act_CNN/` contains the thigh accelerometer CNN functionality 
 - `act_MLP/` contains the thigh accelerometer MLP functionality 
 - `acw_CNN/` contains the wrist accelerometer CNN functionality 
 - `acw_MLP/` contains the thigh accelerometer MLP functionality 
 - `DC_CNN/` contains the depth camera CNN functionality 
 - `PM_CNN/` contains the pressure mat CNN functionality 
 
## Functions
Each module contains the same basic list of functions:
 - `load_ac_attributes_labels(inputPath, verbose=True)`
    - Accepts the path where accelerometer data is stored and a command to show the reading status
    - Extracts and return the accelerometer data and labels
 - `load_DC_images(inputPath, verbose=True)`
    - Accepts the path where depth camera images are stored and a command to show the reading status
    - Extracts and return the depth camera data and labels
 - `load_PM_images(inputPath, verbose=True)`
    - Accepts the path where pressure mat images are stored and a command to show the reading status
    - Extracts and return the pressure mat data and labels
 - `process_ac_attributes(ac_attributes)`
    - Accepts the accelerometers attributes and performs min-max scaling
    - Returns the scaled data
 - `create_cnn()`
    - Creates and returns a CNN model
 - `create_mlp()`
    - Creates and returns an MLP model
 - `create_cnn(height, width, depth, filters=(16, 32, 64))`
    - Accepts data dimensions and filters
    - Implements and returns a CNN model
    
    
    