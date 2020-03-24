from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from PM_CNN import dataset
from PM_CNN import model
import numpy as np
import locale
import os

inputPath = "MEx Dataset/Dataset/pm_1.0_1.0"

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading pressure mat images...")
images, labels = dataset.load_PM_images(inputPath)
# images = images / 255.0

# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(train_images, test_images) = train_test_split(images, test_size=0.30, random_state=42)
(train_labels, test_labels) = train_test_split(labels, test_size=0.30, random_state=42)

# # find the largest house price in the training set and use it to
# # scale our house prices to the range [0, 1] (will lead to better
# # training and convergence)
# maxPrice = trainAttrX["price"].max()
# trainY = trainAttrX["price"] / maxPrice
# testY = testAttrX["price"] / maxPrice

# create our Convolutional Neural Network and then compile the model
# using mean absolute percentage error as our loss, implying that we
# seek to minimize the absolute percentage difference between our
# price *predictions* and the *actual prices*
# model = model.create_cnn(64, 64, 3, regress=True)
model = model.create_cnn(32, 16, 1)
# model = model.create_cnn(12, 16)
# opt = Adam(lr=1e-3, decay=1e-3 / 200)
# model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the model
print("[INFO] training model...")
# model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY),
# 	epochs=200, batch_size=8)
model.fit(train_images, train_labels, epochs=15)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)

# # make predictions on the testing data
# print("[INFO] predicting house prices...")
# preds = model.predict(testImagesX)
#
# # compute the difference between the *predicted* house prices and the
# # *actual* house prices, then compute the percentage difference and
# # the absolute percentage difference
# diff = preds.flatten() - testY
# percentDiff = (diff / testY) * 100
# absPercentDiff = np.abs(percentDiff)
#
# # compute the mean and standard deviation of the absolute percentage
# # difference
# mean = np.mean(absPercentDiff)
# std = np.std(absPercentDiff)
#
# # finally, show some statistics on our model
# locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
# print("[INFO] avg. house price: {}, std house price: {}".format(
# 	locale.currency(df["price"].mean(), grouping=True),
# 	locale.currency(df["price"].std(), grouping=True)))
# print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))