'''
This script was written to create fames to animate the accelerometer data.

imagemagick was used to convert the frames to a gif using a variant ont he following command:
convert -delay 50 *.png accel_animated.gif
'''
import csv

import matplotlib.pyplot as plt

input_file_ac_path = 't'  # input('Chose between wrist and thigh accelerometer [w or t]: ')
if input_file_ac_path == 'w':
    input_file_ac_path = '../dataset/acw'
elif input_file_ac_path == 't':
    input_file_ac_path = '../dataset/act'
# Ask the user the patient number
patient_number = '01'  # input('Enter the patient number (01 - 30): ')
# Ask the user the exercise number
exercise_number = '01_act_1'  # input('Enter the exercise number in the form 0x_act_x or 0x_acw_x (e.g. 01_act_1): ')

# Set graph
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim(-0.8, 1)
ax.set_ylim(-1, 0)
ax.set_zlim(-0.8, -0.2)
ax.view_init(elev=-150, azim=130)
xdata, ydata = [], []
# Gat data from the chosen file and print it on the 3D graph
f = open(f"{input_file_ac_path}/{patient_number}/{exercise_number}.csv")
print(f)
reader = csv.reader(f)
x_vals, y_vals, z_vals = [], [], []

for row in reader:
    x_vals.append(float(row[3]))
    y_vals.append(float(row[2]))
    z_vals.append(float(row[1]))
print(len(x_vals))
# ax.scatter(x_vals, y_vals, z_vals)

step = 20
i, j = 0, 0
while i + step < len(x_vals):
    plt.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.8, 1)
    ax.set_ylim(-1, 0)
    ax.set_zlim(-0.8, -0.2)
    ax.scatter(x_vals[i:i + step], y_vals[i:i + step], z_vals[i:i + step])
    string = 'gif/frame{:05d}.png'.format(j)
    plt.savefig(string)
    i += step
    j += 1
