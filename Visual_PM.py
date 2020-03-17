import numpy
import cv2
import csv
import os

# Initialise vector holding temporary frame
temp_image = numpy.zeros([16, 32])

input_file_pm_path = 'MEx Dataset/Dataset/pm_1.0_1.0'
# Ask the user the patient number
patient_number = input('Enter the patient number (01 - 30): ')
# Ask the user the exercise number
exercise_number = input('Enter the exercise number in the form 0x_pm_x (e.g. 01_pm_1): ')
# Ask the user the folder location of the output video file
output_video_folder = input('Enter the folder location you wish the time lapse video is saved in: ')

# Video Settings
frames_per_seconds = 24.0
save_video_path = f"{output_video_folder}/Timelapse_p{patient_number}_{exercise_number}.mp4"
out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frames_per_seconds, (480, 240))

# Read input file, save frames and put them together in a video
with open(f"{input_file_pm_path}/{patient_number}/{exercise_number}.csv", newline='') as f:
    reader = csv.reader(f)
    name_count = 0
    for row in reader:
        c = 1
        for i in range(31, -1, -1):
            for n in range(0, 16):
                temp_image[n][i] = row[c]
                c = c + 1
        cv2.imwrite(f"frame_{name_count}.jpg", cv2.resize(temp_image / numpy.amax(temp_image) * 256, (480, 240)))

        image_frame = cv2.imread(f"frame_{name_count}.jpg")
        out.write(image_frame)

        os.remove(f"frame_{name_count}.jpg")

        name_count = name_count + 1