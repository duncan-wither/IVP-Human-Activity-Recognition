from sklearn.model_selection import train_test_split
from act_CNN import dataset
from act_CNN import model

inputPath = "MEx Dataset/Dataset/act"

# construct the path to the input .txt file that contains information
print("[INFO] loading thigh accelerometer attributes...")
ac_attributes, ac_labels = dataset.load_ac_attributes_labels(inputPath)

# process the attributes data by performing min-max scaling
print("[INFO] processing data...")
ac_attributesX = dataset.process_ac_attributes(ac_attributes)

# construct a training and testing split with 70% of the data used
# for training and the remaining 30% for evaluation
print("[INFO] constructing training/testing split...")
(train, test) = train_test_split(ac_attributesX, test_size=0.30, random_state=42)
(train_labels, test_labels) = train_test_split(ac_labels, test_size=0.30, random_state=42)

# create an MLP and then compile the model
model = model.create_cnn()
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the model
print("[INFO] training model...")
model.fit(train, train_labels, epochs=50)

test_loss, test_acc = model.evaluate(test, test_labels)

print('\nTest accuracy:', test_acc)