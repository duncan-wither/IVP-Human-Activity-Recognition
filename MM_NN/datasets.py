import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

maxRows = 150


def load_ac_attributes_labels(inputPath, acSensor, verbose=True):
    # maxRows = 150

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
                #     inputPath + '/' + pat_num + '/' + ex_number + '_' + acSensor + '_' + str(p + 1) + '.csv',
                #     header=None, names=cols)
                temp_data = pd.read_csv(
                    inputPath + '/' + pat_num + '/' + ex_number + '_' + acSensor + '_' + str(p + 1) + '.csv',
                    header=None, names=cols, nrows=maxRows)

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


def load_DC_images(inputPath, verbose=True):
    # maxRows = 150

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


def load_PM_images(inputPath, verbose=True):
    # maxRows = 150

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
                #     inputPath + '/' + pat_num + '/' + ex_number + '_pm_' + str(p + 1) + '.csv', header=None).iloc[:,
                #                1:].to_numpy(dtype=float)
                temp_ex_data = pd.read_csv(
                    inputPath + '/' + pat_num + '/' + ex_number + '_pm_' + str(p + 1) + '.csv', header=None,
                    nrows=maxRows).iloc[:, 1:].to_numpy(dtype=float)
                # Convert the dat in the format ndarray(# of rows, 32, 16)
                data = np.zeros((len(temp_ex_data), 32, 16, 1), dtype=float)
                for r in range(len(temp_ex_data)):
                    c = 0
                    for d in range(31, -1, -1):
                        for e in range(0, 16):
                            data[r][d][e][0] = temp_ex_data[r][
                                c]  # Contains the full exercise file in the form ndarray(# of rows, 12,16)
                            c = c + 1
                # If it is the first iteration initialise array and copy value
                if first_iteration == True:
                    # Images
                    images = np.zeros((len(data), 32, 16, 1), dtype=float)
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
