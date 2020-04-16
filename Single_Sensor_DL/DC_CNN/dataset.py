# import the necessary packages
import numpy as np
import pandas as pd

# import cv2

maxRows = 150


# def load_DC_attributes(inputPath):
#     # initialize the list of column names in the CSV file and then
#     # load it using Pandas
#     cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
#     df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
#
#     # determine (1) the unique zip codes and (2) the number of data
#     # points with each zip code
#     zipcodes = df["zipcode"].value_counts().keys().tolist()
#     counts = df["zipcode"].value_counts().tolist()
#
#     # loop over each of the unique zip codes and their corresponding
#     # count
#     for (zipcode, count) in zip(zipcodes, counts):
#         # the zip code counts for our housing dataset is *extremely*
#         # unbalanced (some only having 1 or 2 houses per zip code)
#         # so let's sanitize our data by removing any houses with less
#         # than 25 houses per zip code
#         if count < 25:
#             idxs = df[df["zipcode"] == zipcode].index
#             df.drop(idxs, inplace=True)
#
#     # return the data frame
#     return df
#
# def process_house_attributes(df, train, test):
#     # initialize the column names of the continuous data
#     continuous = ["bedrooms", "bathrooms", "area"]
#
#     # performin min-max scaling each continuous feature column to
#     # the range [0, 1]
#     cs = MinMaxScaler()
#     trainContinuous = cs.fit_transform(train[continuous])
#     testContinuous = cs.transform(test[continuous])
#
#     # one-hot encode the zip code categorical data (by definition of
#     # one-hot encoing, all output features are now in the range [0, 1])
#     zipBinarizer = LabelBinarizer().fit(df["zipcode"])
#     trainCategorical = zipBinarizer.transform(train["zipcode"])
#     testCategorical = zipBinarizer.transform(test["zipcode"])
#
#     # construct our training and testing data points by concatenating
#     # the categorical features with the continuous features
#     trainX = np.hstack([trainCategorical, trainContinuous])
#     testX = np.hstack([testCategorical, testContinuous])
#
#     # return the concatenated training and testing data
#     return (trainX, testX)

# def load_house_images(df, inputPath):
def load_DC_images(inputPath, verbose=True):
    first_iteration = True

    # Iterate through the training patients
    # for i in range(training_patients_number):
    for i in range(30):
        pat_num = '{:0>2d}'.format(i + 1)
        if verbose:
            print('Reading patient ' + pat_num + ' data...')
        # Iterate through the 8 exercises
        for n in range(7):
            ex_number = '{:0>2d}'.format(n + 1)
            # Consider that the 4th exercise has 2 files
            if ex_number == '04':
                LR_num = 2
            else:
                LR_num = 1
            for p in range(LR_num):
                # Extract tha data from the .csv file in the format ndarray(# of rows, 192)
                # temp_ex_data = pd.read_csv(
                #     inputPath + '/' + pat_num + '/' + ex_number + '_dc_' + str(p + 1) + '.csv', header=None).iloc[:,
                #                1:].to_numpy(dtype=float)
                temp_ex_data = pd.read_csv(
                    inputPath + '/' + pat_num + '/' + ex_number + '_dc_' + str(p + 1) + '.csv', header=None,
                    nrows=maxRows).iloc[:, 1:].to_numpy(dtype=float)
                # Convert the dat in the format ndarray(# of rows, 12, 16)
                # data = np.zeros((len(temp_ex_data), 12, 16), dtype=float)
                data = np.zeros((len(temp_ex_data), 12, 16, 1), dtype=float)
                for r in range(len(temp_ex_data)):
                    c = 0
                    for d in range(11, -1, -1):
                        for e in range(0, 16):
                            data[r][d][e][0] = temp_ex_data[r][
                                c]  # Contains the full exercise file in the form ndarray(# of rows, 12,16)
                            c = c + 1
                # If it is the first iteration initialise array and copy value
                if first_iteration == True:
                    # Images
                    # images = np.zeros((len(data), 12, 16), dtype=float)
                    images = np.zeros((len(data), 12, 16, 1), dtype=float)
                    images = data

                    # Labels
                    labels = np.zeros(len(data))
                    if LR_num == 2:
                        labels = np.ones(len(data)) * 7
                    else:
                        labels = np.ones(len(data)) * (int(ex_number) - 1)

                    first_iteration = False
                # Else append to existing array
                else:
                    # Images
                    images = np.append(images, data, 0)

                    # Labels
                    if LR_num == 2:
                        labels = np.append(labels, np.ones(len(data)) * 7)
                    else:
                        labels = np.append(labels, np.ones(len(data)) * (int(ex_number) - 1))

    return images, labels
