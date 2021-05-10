# MNIST

MNIST has 60000 training and 10000 testing datasets which are 28x28 grayscale images.
In my observation, the model can be trained effectively and efficiently using the
following model architecture:
I’ve chosen a Sequential model for this classification. In addition to the model I’ve used 2
Convolutional layers of 32 filters, 2 Convolutional layers of 64 filters, along with Max
pooling layers of pool size (2x2) and Dropout layers to avoid overfitting a Flatten layer to
convert the 2 dimensional matrix of features into a vector that can be passed into a fully
connected layer i.e., the Dense layer of 512 neurons followed by an output layer, a Dense
layer of 10 neurons to classify the 10 numbers [0-9].
