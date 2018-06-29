### Overview

Training a Convolutional Neural Network with only numpy.

### Comment

The architecture of the code is similar to caffe. In each kind of layer, we need to implement the forward and backward method. Extra memory is allocated to store the bottom_data and top_diff of each layer. The convolutional operator is represented as matrix multiplation, though the the matrix is not that sparse and the image is represented as a 1d vector instead of 2d matrix.

### Dependency

You need download mnist dataset and specify the directory in the command line arguments.
