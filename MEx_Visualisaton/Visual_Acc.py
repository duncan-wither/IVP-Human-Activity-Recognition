import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ask the user to chose between thigh or wrist accelerometer
input_file_ac_path = input('Chose between wrist and thigh accelerometer [w or t]: ')
if input_file_ac_path == 'w':
    input_file_ac_path = '../MEx Dataset/Dataset/acw'
elif input_file_ac_path == 't':
    input_file_ac_path = '../MEx Dataset/Dataset/act'
# Ask the user the patient number
patient_number = input('Enter the patient number (01 - 30): ')
# Ask the user the exercise number
exercise_number = input('Enter the exercise number in the form 0x_act_x or 0x_acw_x (e.g. 01_act_1): ')

# Set graph
fig = plt.figure()
ax = plt.axes(projection='3d')

# Gat data from the chosen file and print it on the 3D graph
with open(f"{input_file_ac_path}/{patient_number}/{exercise_number}.csv", newline='') as f:
    reader = csv.reader(f)
    firstIteration = True
    # Start at the firs point of the dataset
    for row in reader:
        if firstIteration == True:
            prevXorigin = float(row[1])
            prevYorigin = float(row[2])
            prevZorigin = float(row[3])

            firstIteration = False
        # Draw a line between the previus and current point
        else:
            xorigin = float(row[1])
            yorigin = float(row[2])
            zorigin = float(row[3])

            zline = ([prevXorigin, xorigin])
            xline = ([prevYorigin, yorigin])
            yline = ([prevZorigin, zorigin])
            ax.plot3D(xline, yline, zline, 'blue')

            prevXorigin = xorigin
            prevYorigin = yorigin
            prevZorigin = zorigin

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
