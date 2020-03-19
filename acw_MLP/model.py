from keras.models import Sequential
from keras.layers.core import Dense

def create_mlp(dim):
    # define our MLP network
    model = Sequential()
    model.add(Dense(3, input_dim=dim, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(11, activation="relu"))
    model.add(Dense(13, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(17, activation="relu"))
    model.add(Dense(19, activation="relu"))
    model.add(Dense(17, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(13, activation="relu"))
    model.add(Dense(11, activation="relu"))
    model.add(Dense(8, activation="softmax"))

    return model