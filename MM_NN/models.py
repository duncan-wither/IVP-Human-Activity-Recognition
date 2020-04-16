from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential


def create_act_mlp(dim):
    # define our MLP network
    model = Sequential()
    model.add(Dense(3, input_dim=dim, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(11, activation="relu"))
    model.add(Dense(13, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(13, activation="relu"))
    model.add(Dense(11, activation="relu"))
    model.add(Dense(8, activation="softmax"))

    return model


def create_acw_mlp(dim):
    # define our MLP network
    model = Sequential()
    model.add(Dense(3, input_dim=dim, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(11, activation="relu"))
    model.add(Dense(13, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(13, activation="relu"))
    model.add(Dense(11, activation="relu"))
    model.add(Dense(8, activation="softmax"))

    return model


def create_DC_cnn(height, width, depth, filters=(16, 32, 64)):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
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
    x = Dense(8)(x)
    x = Activation("softmax")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model


def create_PM_cnn(height, width, depth, filters=(16, 32, 64)):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
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
    x = Dense(8)(x)
    x = Activation("softmax")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model
