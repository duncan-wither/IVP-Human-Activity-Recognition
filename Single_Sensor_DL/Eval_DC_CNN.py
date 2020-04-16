from sklearn.model_selection import train_test_split

from DC_CNN import dataset
from DC_CNN import model

inputPath = "dataset/dc_0.05_0.05"

print("[INFO] loading depth camera images...")
images, labels = dataset.load_DC_images(inputPath)


# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(train_images, test_images) = train_test_split(images, test_size=0.30, random_state=42)
(train_labels, test_labels) = train_test_split(labels, test_size=0.30, random_state=42)

# create our Convolutional Neural Network and then compile the model
# using mean absolute percentage error as our loss.
model = model.create_cnn(12, 16, 1)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the model
print("[INFO] training model...")
# model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY),
# 	epochs=200, batch_size=8)
model.fit(train_images, train_labels, epochs=15)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)
