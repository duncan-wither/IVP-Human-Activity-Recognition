import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

maxRows = 150


def load_ac_attributes_labels(inputPath, verbose=True):
    first_iteration = True

    # initialize the list of column names in the CSV file and then
    # load it using Pandas
    cols = ["x", "y", "z"]

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
                # Extract tha data from the .csv file in the format ndarray(# of rows, 3)
                # temp_data = pd.read_csv(
                #     inputPath + '/' + pat_num + '/' + ex_number + '_act_' + str(p + 1) + '.csv', header=None,
                #     names=cols)
                temp_data = pd.read_csv(
                    inputPath + '/' + pat_num + '/' + ex_number + '_act_' + str(p + 1) + '.csv', header=None,
                    names=cols, nrows=maxRows)

                if first_iteration == True:
                    # Data
                    data = pd.DataFrame({cols[0]: [],
                                         cols[1]: [],
                                         cols[2]: []})
                    data = data.append(temp_data)

                    # Labels
                    labels = np.zeros(len(temp_data))
                    if p == 1:
                        labels = np.ones(len(temp_data)) * 7
                    else:
                        labels = np.ones(len(temp_data)) * (int(ex_number) - 1)

                    first_iteration = False
                # Else append to existing array
                else:
                    # Data
                    data = data.append(temp_data)

                    # Labels
                    if LR_num == 2:
                        labels = np.append(labels, np.ones(len(temp_data)) * 7)
                    else:
                        labels = np.append(labels, np.ones(len(temp_data)) * (int(ex_number) - 1))

    # return the data frame
    return data, labels


# def process_house_attributes(df, train, test)
def process_ac_attributes(train, test):
    # initialize the column names of the continuous data
    continuous = ["x", "y", "z"]

    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainX = cs.fit_transform(train[continuous])
    testX = cs.transform(test[continuous])

    # return the concatenated training and testing data
    return (trainX, testX)
