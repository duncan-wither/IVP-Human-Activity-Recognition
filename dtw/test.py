#!/usr/bin/env python
''' test.py
Created by slam at 07/02/2020

Desctiption:

'''
# LIBS
## Default Libs
import pickle

## 3rd Party Libs
import numpy as np
import pandas as pd
import cv2  # opencv-python

## Pathfinding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

## Custom Libs
from dtw import *


'''
TODO
*1 - Create DTW function w/ two potential datasets
*    1a - create map of the euclideian distances at each time between the two images
*        1ai - Use the sum of the Euclian distances between the pixels
*    1b - find shortest path from the start of both to the end of both
*        1bi - no reverse time in either.
         1bii- could be ways to speed this up
*    1c - sum? the distances of the paths. Use this as the Distance Algorithm
 2 - Create simple kNN algorithm

Biblio:
https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
https://medium.com/@shachiakyaagba_41915/dynamic-time-warping-with-time-series-1f5c05fb8950
https://cran.r-project.org/web/packages/tsfknn/vignettes/tsfknn.html
'''

# Reading Depth camera data
# Var naming is patient#exercise#
# Data is col 1 = time, col 2-193 is the 16x12 camera feed
# So read the data, ignore the first column and convert to numpy array w/ type float
p1ex1 = pd.read_csv('../MEx Dataset/Dataset/dc_0.05_0.05/01/01_dc_1.csv').iloc[:, 1:].to_numpy(dtype=float)
p1ex2 = pd.read_csv('../MEx Dataset/Dataset/dc_0.05_0.05/01/02_dc_1.csv').iloc[:, 1:].to_numpy(dtype=float)
p2ex1 = pd.read_csv('../MEx Dataset/Dataset/dc_0.05_0.05/02/01_dc_1.csv').iloc[:, 1:].to_numpy(dtype=float)
# converted to np array as DataFrames appeared to be slow and cumbersome to use, not to mention superfluous.


calc_MAP = True
# creating an array for the map
# rows is p1, cols is p2
if calc_MAP:
    MAP1 = create_map(p1ex1, p2ex1)
    print('Found Map 1')
    MAP2 = create_map(p1ex2, p2ex1)
    print('Found Map 2')
    
    # Saving MAPs
    f = open('MAP.pckl', 'wb')
    pickle.dump([MAP1, MAP2], f)
    f.close()

else:
    # Retriveing MAPS
    f = open('MAP.pckl', 'rb')
    MAP1, MAP2 = pickle.load(f)
    f.close()

# converting map to image
MAP1_img = 255 * (MAP1 / MAP1.max())
MAP1_img[len(p1ex1) - 1, 0] = 1  # Start Point
MAP1_img[0, len(p2ex1) - 1] = 1  # End Point
cv2.imwrite('DTW1.png', MAP1_img)

MAP2_img = 255 * (MAP2 / MAP2.max())
cv2.imwrite('DTW2.png', MAP2_img)

# Pathfinding

find_path = True

if find_path:
    
    path1 = dtw_path(MAP1)
    print('Found Path 1')
    path2 = dtw_path(MAP2)
    print('Found Path 2')
    
    f = open('PATH.pckl', 'wb')
    pickle.dump([path1, path2], f)
    f.close()

else:
    f = open('PATH.pckl', 'rb')
    path1, path2 = pickle.load(f)
    f.close()

# Adding the path to the image
for i in range(len(path1)):
    MAP1_img[path1[i][1], path1[i][0]] = 0
cv2.imwrite('DTW1_path.png', MAP1_img)

for i in range(len(path2)):
    MAP2_img[path2[i][1], path2[i][0]] = 0
cv2.imwrite('DTW2_path.png', MAP2_img)

# calulating the DTW path cost
DTW_cost_1 = 0
for i in range(len(path1)):
    DTW_cost_1 += MAP1[path1[i][1], path1[i][0]]
print('DTW cost 1 = ', DTW_cost_1)

DTW_cost_2 = 0
for i in range(len(path2)):
    DTW_cost_2 += MAP2[path2[i][1], path2[i][0]]

print('DTW cost 2 = ', DTW_cost_2)

https://www.findaphd.com/phds/project/robotic-inspection-of-complex-welds/?p109320
https://trello.com/b/jid4j362/project-general-board
https://app.slack.com/client/TNKDU3CJU/CPZDG2ZJT/thread/CPJV12MSS-1580761455.001100
https://docs.google.com/document/d/14FJMkg3oPpw017UU1mIQOixZ1o-s_GIG5GH7TgDEhqM/edit#heading=h.q2l60vi08yje
https://drive.google.com/drive/folders/1a6UFh1Qc5175VRSrDJDirhQjOFLrCKcK?ths=true
https://docs.google.com/document/d/13jFSF6c9UNDy_LR0PNbxswz6tiQ_sCuVnkp52LwMAvM/edit
https://www.onsemi.com/pub/Collateral/LM2576-D.PDF
https://www.google.com/search?sxsrf=ACYBGNSU11oGJESTqrWGVKhoBcpw_SQrkQ%3A1580994529826&ei=4Q88Xp_zMbqk1fAP-t28wAw&q=pmw+controlled+buck+converter&oq=pmw+controlled+buck+converter&gs_l=psy-ab.3..0i13.35270.43962..44893...17.1..0.99.2919.41......0....1..gws-wiz.......0i71j0i8i7i30j0i8i30j0i8i13i30j0i7i30j0i13i5i30.TabPZ8dxDVU&ved=0ahUKEwjf593K_7znAhU6UhUIHfouD8gQ4dUDCAs&uact=5
https://www.bitsbox.co.uk/index.php?main_page=product_info&products_id=3202
http://www.ti.com/product/LM5145?utm_source=google&utm_medium=cpc&utm_campaign=APP-BSR-null-prodfolderdynamic-cpc-pf-google-eu&utm_content=prodfolddynamic&ds_k=DYNAMIC+SEARCH+ADS&DCM=yes&gclid=CjwKCAiAj-_xBRBjEiwAmRbqYp9g5cK7zhP4RUpGMNH6DpImi6PHFx6Q-N6fX9850d0R4uAKFO96nRoCEbQQAvD_BwE&gclsrc=aw.ds
https://forum.arduino.cc/index.php?topic=517260.0
http://www.yourduino.com/sunshop/index.php?l=product_detail&p=522
https://www.researchgate.net/post/How_control_the_output_of_a_DC_buck_converter
https://www.we-online.com/catalog/en/pm/step-down-converter/magic_variable_output_voltage/
https://docs.google.com/document/d/1Yp5eph-4mj9pXCbTYFPCHL3uy0ggDQWlmA1EYnycV4U/edit
https://www.google.com/search?q=esc+satasheet&oq=esc+satasheet&aqs=chrome..69i57j0l7.2644j0j7&sourceid=chrome&ie=UTF-8
file:///home/slam/Downloads/30A_BLDC_ESC_Product_Manual.pdf
https://hobbyking.com/media/file/1006513628X70599X0.pdf
https://www.first4magnets.com/magnet-materials-standard-assemblies-c150/19-05mm-dia-x-31-75mm-thick-electromagnet-with-5mm-mounting-hole-4-5kg-pull-12v-dc-2w-p12593#ps_0_13328|ps_1_13270
https://drive.google.com/drive/folders/1a6UFh1Qc5175VRSrDJDirhQjOFLrCKcK?ths=true
https://www.draw.io/?state=%7B%22folderId%22:%221a6UFh1Qc5175VRSrDJDirhQjOFLrCKcK%22,%22action%22:%22create%22,%22userId%22:%22110014531197310169968%22%7D#G1BRae0yXlvG4neD8YM1SyvCzbTDZBfLFX
https://www.instructables.com/id/How-to-use-OV7670-Camera-Module-with-Arduino/
https://uk.mail.yahoo.com/d/folders/1
https://github.com/duncan-wither/advanced-systems-em502
