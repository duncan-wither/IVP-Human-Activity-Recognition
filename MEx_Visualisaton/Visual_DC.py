import csv
import os

import cv2
import numpy

# Initialise vector holding temporary frame
temp_image = numpy.zeros([12, 16])

input_file_dc_path = '../dataset/dc_0.05_0.05'
# Ask the user the patient number
patient_number = '05'  # input('Enter the patient number (01 - 30): ')
# Ask the user the exercise number
exercise_number = '01_dc_1'  # input('Enter the exercise number in the form 0x_dc_x (e.g. 01_dc_1): ')
# Ask the user the folder location of the output video file
output_video_folder = ''  # input('Enter the folder location you wish the time lapse video is saved in: ')

# Video Settings
frames_per_seconds = 24.0
save_video_path = 'test.mp4'  # f"{output_video_folder}/Timelapse_p{patient_number}_{exercise_number}.mp4"
out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frames_per_seconds, (320, 240))

# Read input file, save frames and put them together in a video
with open(f"{input_file_dc_path}/{patient_number}/{exercise_number}.csv", newline='') as f:
    reader = csv.reader(f)
    name_count = 0
    for row in reader:
        c = 1
        for i in range(11, -1, -1):
            for n in range(0, 16):
                temp_image[i][n] = row[c]
                c = c + 1
        cv2.imwrite(f"frame_{name_count}.jpg",
                    cv2.resize(temp_image * 256, (320, 240), interpolation=cv2.INTER_NEAREST))

        image_frame = cv2.imread(f"frame_{name_count}.jpg")
        out.write(image_frame)

        os.remove(f"frame_{name_count}.jpg")

        name_count = name_count + 1
