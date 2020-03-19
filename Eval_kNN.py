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
import numpy as np

# Custom Libs
import knn

# 3rd Party Libs

# Constants
RUNS = 3
TESTING_PERCENT = 70 / 100
EXERCISES = [1, 2, 3, 4, 5, 6, 7, 8]
NO_EX = len(EXERCISES)  # No of Exercises
NO_PAT = 30
EVAL_NEEDED = False  # True  # Used if the Evaluation hasn't already occured

# initialising save list
eval_list = []
# eval_list[Run#][sens][k_val](act_ex,guess)


if EVAL_NEEDED:
    runs_for_eval = RUNS
else:
    # f = open('Eval_Matrix.pckl', 'wb')  # sorting the sorted costs
    f = open('Eval_Matrix.pckl', 'rb')
    eval_list = pickle.load(f)
    f.close()
    # eval_list has heirarchy [run#][0][sens#][0][test_no][0][iteration][trip#]
    # test no is just one of the exercises in the testing list
    # iteration is one of the 9 times it runs per exercise
    # trip is the (execise, guessed exercise, time taken)
    runs_for_eval = 0

i = 0
for run_no in range(runs_for_eval):
    
    # Initialising Lists
    traning_list = []
    testing_list = []
    run_eval_list = []
    
    # Creating Sets for testing and training
    # total list of paitients and exercises
    patex_list = rd.sample(range(NO_PAT * NO_EX), int(NO_PAT * NO_EX * TESTING_PERCENT))
    for pat in range(NO_PAT):
        for ex in range(NO_EX):
            if (ex + NO_EX * pat) in patex_list:
                traning_list.append((pat + 1, ex + 1))
            else:
                testing_list.append((pat + 1, ex + 1))
    
    # Shuffling Lists
    rd.shuffle(traning_list)
    rd.shuffle(testing_list)
    
    # Running KNN
    # for each sensor
    for sens in ['act', 'acw', 'dc', 'pm']:
        
        # Setting the Down sampling Rate.
        if sens == 'act' or sens == 'acw':
            dsr = 10
        else:
            dsr = 1
        
        NN_sens_list = []
        for test_pair in testing_list:
            t0 = time.time()
            costs = knn.find_costs(traning_list, test_pair, down_sample_rate=dsr, verbose=False, sens_str=sens)
            t1 = time.time()
            
            # presenting a % done
            i += 1
            print(100 * i / (RUNS * 4 * len(testing_list)), '% Done in ', t1 - t0, ' Seconds.')
            NN_list = []
            for k_val in range(1, 10):
                NN_list.append([test_pair[1], knn.pick_nn(costs, k_val, verbose=False), t0 - t1, costs])
            NN_sens_list.append([NN_list])
        run_eval_list.append([NN_sens_list])
    
    eval_list.append([run_eval_list])
    
    f = open('Eval_Matrix.pckl', 'wb')  # sorting the sorted costs
    pickle.dump(eval_list[0], f)
    f.close()

# Evaluating the Results

# Looking at K_values
t_ave = 0
legend_labels = ['Thigh Accel.', 'Wrist Accel.', 'Depth Camera', 'Pressure Mat']
for sens in range(4):
    k_acc = []  # creating an empty list of accuracies for this data
    k_val = []
    for k_ind in range(9):
        # iterating over all possible k values (kNN).
        acc = 0  # setting current accuracy to zero
        for run in range(RUNS):
            
            for i in range(72):
                trip = eval_list[run][0][sens][0][i][0][
                    k_ind]  # reading the trpiple (execise, guessed exercise, time taken)
                # print(run, sens, i, k_ind)
                t_ave -= trip[2]  # keeps a running total of the time taken
                if trip[0] == trip[1]:
                    acc += 1
        
        p_acc = 100 * acc / (RUNS * 72)
        print(legend_labels[sens], ' using a k=', k_ind + 1, ' has accuracy of ', str(p_acc), '%')
        k_acc.append(p_acc)
        k_val.append(k_ind + 1)
    
    # Plot results depending on k
    plt.plot(k_val, k_acc, label=legend_labels[sens])

t_ave /= RUNS * 9 * 63 * 4
print('Average Time Taken', t_ave)
plt.title('Accuracy when comparing different Neighbors')
plt.xlabel('No. of nearest neighbors Considered')
plt.ylabel('Accuracy (%)')
plt = plt
plt.legend(framealpha=1.0)
plt.savefig('Accuracy_vs_K.png')

# use the results from that experiment to decide on the final evaluation:
# Confidence values:
# act_conf = 0.638  # k=1
# acw_conf = 0.329  # k=4 or 5
# dc_conf = 0.565  # k=4
# pm_conf = 0.384  # k-2

# Sensor Confidence in indexible form
# [act,acw,dc,pm]
conf_vales = [0.638, 0.329, 0.565, 0.384]
best_k_val = [1, 4, 4, 2]

# lsits to hold the accuracy of each exercise
# [act,acw,dc,pm,combined]
accuracies = np.zeros((5, 1))
for run in range(RUNS):
    for i in range(72):
        # to hold the multimodal result
        mm_res = np.zeros((8, 1))
        actual_ex = eval_list[run][0][0][0][i][0][1][0]  # as the actual_ex (k and sens values shouldnt mattter)
        for sens in range(4):
            k_ind = best_k_val[sens] - 1
            trip = eval_list[run][0][sens][0][i][0][k_ind]
            # actual_ex = trip[0] #this shouldn't change from sensor to sensor
            est_ex = trip[1]  # estimated exercise
            # Check the results
            if trip[0] == trip[1]:
                accuracies[sens, 0] += 1
            # for the combined result
            mm_res[est_ex - 1, 0] += conf_vales[sens]  # provides weighting based on confidence
        
        final_ex = np.argmax(mm_res) + 1
        if final_ex == actual_ex:
            accuracies[4, 0] += 1

print()
for i in range(4):
    print(legend_labels[i], ' using a k=', best_k_val[i], ' has accuracy of ', accuracies[i, 0] / (RUNS * 72 * 0.01),
          '%')
final_acc = accuracies[4, 0] / (RUNS * 72 * 0.01)
print('The Combined Estimate has accuracy of ', accuracies[4, 0] / (RUNS * 72 * 0.01), '%')

plt.plot([1, 9], [final_acc, final_acc], label='Combined Result')
plt.plot([1, 9], [100 / 8, 100 / 8], label='Random Choice')

plt.legend(loc="right", framealpha=1.0)
plt.savefig('Accuracy_vs_K_and_Final.png')
