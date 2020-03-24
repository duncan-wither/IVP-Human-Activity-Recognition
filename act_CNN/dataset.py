from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

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
                # Extract tha data from the .csv file in the format ndarray(maxRows, 3)
                temp_data = pd.read_csv(
                    inputPath + '/' + pat_num + '/' + ex_number + '_act_' + str(p + 1) + '.csv', header=None,
                    names=cols, nrows=maxRows)

                if first_iteration == True:
                    # Data
                    data = pd.DataFrame({cols[0]: [],
                                         cols[1]: [],
                                         cols[2]: []})
                    data = data.append(temp_data)

                    first_iteration = False
                # Else append to existing array
                else:
                    # Data
                    data = data.append(temp_data)

    l = [0, 1, 2, 3, 7, 4, 5, 6]
    labels = np.zeros(240, dtype=int)
    for i in range(240):
        labels[i] = l[i % 8]

    # return the data frame
    return data, labels

# def process_house_attributes(df, train, test)
def process_ac_attributes(ac_attributes):
    # initialize the column names of the continuous data
    continuous = ["x", "y", "z"]

    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    ac_attributesX = cs.fit_transform(ac_attributes[continuous])

    # Convert data in the rotm ndarray(240, 150*3)
    ac_attributes_reshaped = np.zeros((240, 450), dtype=float)
    tempX = np.zeros(150, dtype=float)
    tempY = np.zeros(150, dtype=float)
    tempZ = np.zeros(150, dtype=float)
    c = 0
    for n in range(len(ac_attributesX)):
        tempX[n % 150] = ac_attributesX[n][0]
        tempY[n % 150] = ac_attributesX[n][1]
        tempZ[n % 150] = ac_attributesX[n][2]

        if (n % 150) == 149:
            tempData = np.append(tempX, tempY)
            tempData = np.append(tempData, tempZ)

            ac_attributes_reshaped[c][:] = tempData

            c = c + 1

    # return the concatenated training and testing data
    return ac_attributes_reshaped