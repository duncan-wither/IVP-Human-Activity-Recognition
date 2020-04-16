# import the necessary packages
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model


# def create_mlp(dim, regress=False):
# #     # define our MLP network
# #     model = Sequential()
# #     model.add(Dense(8, input_dim=dim, activation="relu"))
# #     model.add(Dense(4, activation="relu"))
# #
# #     # check to see if the regression node should be added
# #     if regress:
# #         model.add(Dense(1, activation="linear"))
# #
# #     # return our model
# #     return model

# def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
# def create_cnn(width, height, depth, filters=(16, 32, 64)):
def create_cnn(height, width, depth, filters=(16, 32, 64)):
    # def create_cnn(width, height, filters=(16, 32, 64)):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    # inputShape = (height, width)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    # x = Dense(4)(x)
    x = Dense(8)(x)
    # x = Activation("relu")(x)
    x = Activation("softmax")(x)

    # # check to see if the regression node should be added
    # if regress:
    #     x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model
