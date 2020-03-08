#!/usr/bin/env python
""" Eval_kNN.py
Created by slam at 08/03/2020

Description: Runs both the MC-DCNN and the kNN with DTW methods multiple times to get accuracy and time taken
"""

import pickle
# Default Libs
import random as rd
import time

import matplotlib.pyplot as plt

# Custom Libs
import knn

# 3rd Party Libs

# Constants
RUNS = 2
TESTING_PERCENT = 70 / 100
EXERCISES = [1, 2, 3, 4, 5, 6, 7]
EVAL_NEEDED = True  # Used if the Evaluation hasn't already occured

# initialising save list
eval_list = []
# eval_list[Run#][sens][k_val](act_ex,guess)


if EVAL_NEEDED:
    runs_for_eval = RUNS
else:
    f = open('Eval_Matrix.pckl', 'wb')  # sorting the sorted costs
    eval_list = pickle.load(f)
    f.close()
    runs_for_eval = 0

i = 0
for run_no in range(runs_for_eval):
    
    # Initialising Lists
    traning_list = []
    testing_list = []
    run_eval_list = []
    
    # Creating Sets for testing and training
    # total list of paitients and exercises
    patex_list = rd.sample(range(210), int(210 * TESTING_PERCENT))
    for pat in range(30):
        for ex in range(7):
            if (ex + 7 * pat) in patex_list:
                traning_list.append((pat + 1, ex + 1))
            else:
                testing_list.append((pat + 1, ex + 1))
    
    # Shuffling Lists
    rd.shuffle(traning_list)
    rd.shuffle(testing_list)
    
    # Running KNN
    # for each sensor
    for sens in ['act', 'acw', 'dc', 'pm']:
        NN_sens_list = []
        for test_pair in testing_list:
            t0 = time.time()
            costs = knn.find_costs(traning_list, test_pair, verbose=0, down_sample_rate=100, sens_str=sens)
            t1 = time.time()
            
            # presenting a % done
            i += 1
            print(100 * i / (RUNS * 4 * len(testing_list)), '% Done')
            NN_list = []
            for k_val in range(1, 10):
                NN_list.append([test_pair[1], knn.pick_nn(costs, k_val, verbose=False), t0 - t1, costs])
            NN_sens_list.append([NN_list])
        run_eval_list.append([NN_sens_list])
    
    eval_list.append([run_eval_list])
    
    f = open('Eval_Matrix.pckl', 'wb')  # sorting the sorted costs
    pickle.dump(eval_list, f)
    f.close()

# Evaluating the Results

# Looking at K_values
t_ave = 0
legend_labels = ['Thigh Accel.', 'Wrist Accel.', 'Depth Camera', 'Pressure Mat']
for sens in range(4):
    k_acc = []
    for k in range(9):
        acc = 0
        for run in range(RUNS):
            for i in range(63):
                trip = eval_list[run][sens][k][i]
                t_ave += trip[2]
                if trip[0] == trip[1]:
                    acc += 1
        k_acc.append(acc / (RUNS * 4 * 64))
        
        # Plot results depending on k
        plt.plot(k_acc, label=legend_labels[sens])

t_ave /= RUNS * 9 * 63 * 4
print('Average Time Taken', t_ave)
plt.title('Accuracy when comparing different Neighbors')
plt.xlabel('No. of nearest neighbors Considered')
plt.ylabel('Accuracy (%)')
plt.savefig('Accuracy_vs_K.png')

'''
https://trello.com/b/tRZwdjFc/life-organisation
https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480
https://github.com/duncan-wither?tab=repositories
https://www.google.com/search?q=cnn+architecture&oq=CNN+arch&aqs=chrome.0.0j69i57j0l6.1971j0j7&sourceid=chrome&ie=UTF-8
https://www.w3schools.com/python/ref_random_shuffle.asp
https://thispointer.com/python-how-to-check-if-an-item-exists-in-list-search-by-value-or-condition/
https://matplotlib.org/tutorials/intermediate/legend_guide.html
https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/
https://link.springer.com/chapter/10.1007/11847465_11
https://link.springer.com/chapter/10.1007/11548706_13
'''
