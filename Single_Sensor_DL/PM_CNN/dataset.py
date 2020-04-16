import pandas as pd
import pandas as pd
import numpy as np
# import cv2

maxRows = 150


# def load_house_images(df, inputPath):
def load_PM_images(inputPath, verbose=True):
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
                # data = np.zeros((len(temp_ex_data), 12, 16), dtype=float)
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
                    # images = np.zeros((len(data), 12, 16), dtype=float)
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
