from sklearn.model_selection import train_test_split

from act_MLP import dataset
from act_MLP import model

input_path = "dataset/act"

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading thigh accelerometer attributes...")
ac_attributes, ac_labels = dataset.load_ac_attributes_labels(input_path)

# construct a training and testing split with 70% of the data used
# for training and the remaining 30% for evaluation
print("[INFO] constructing training/testing split...")
(train, test) = train_test_split(ac_attributes, test_size=0.30, random_state=42)
(train_labels, test_labels) = train_test_split(ac_labels, test_size=0.30, random_state=42)

# process the house attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
print("[INFO] processing data...")
(trainX, testX) = dataset.process_ac_attributes(train, test)

# create our MLP and then compile the model using mean absolute
# percentage error as our loss, implying that we seek to minimize
# the absolute percentage difference between our price *predictions*
# and the *actual prices*
model = model.create_mlp(trainX.shape[1])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the model
print("[INFO] training model...")
model.fit(trainX, train_labels, epochs=15)

test_loss, test_acc = model.evaluate(testX, test_labels)

print('\nTest accuracy:', test_acc)
