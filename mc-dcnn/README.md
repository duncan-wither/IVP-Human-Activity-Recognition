# Multi-Channel Deep Convolution Neural Network (MC-DCNN)
This module aims to replicate the functionality shown in 
[this paper](http://link.springer.com/10.1007/978-3-319-08010-9_33) by Zheng et. al.

### How it works
 1. For each channel of multivariate data use a single channel of the DCNN
 2. Combine in a multi-layered perceptron
 
### Todo's
 - [ ] Create DCNN
    - [ ] Create DCNN for one dataset (act in this case)
      - [x] Read data to one suitable for tensorflow (tf)
        - [ ] Variable Length datasets?
      - [ ] Understand the `Conv1D` filter inputs
      - [ ] Get a model to fit.
      - [ ] Produce some form of result for a 'test case'
    - [ ] Transpose this for each dataset
    - [ ] Combine into a module
 - [ ] Create multi-layered perceptron (MLP)
 - [ ] Combine Data in MLP