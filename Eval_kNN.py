#!/usr/bin/env python
""" Eval_kNN.py
Created by slam at 08/03/2020

Description: Runs both the kNN with DTW methods multiple times to get accuracy and time taken
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
RUNS = 6  # 3
TRAINING_PERCENT = 70 / 100
EXERCISES = [1, 2, 3, 4, 5, 6, 7, 8]
NO_PAT = 30
EVAL_NEEDED = True  # Used if the Evaluation hasn't already occurred
PCKL_STR = 'Eval_Matrix.pckl'  # Where to save results or retrieve stored results
K_VAL_ANALYSIS = False  # True performs the analysis the range of k-Values
UPDATE_FIGS = True  # bool to update the saved figures

# Derived Constants
NO_EX = len(EXERCISES)  # No of Exercises
TESTING_NO = 5

# initialising save list
eval_list = []
# eval_list[Run#][sens][k_val](act_ex,guess)


if EVAL_NEEDED:
    runs_for_eval = RUNS
else:
    # f = open('Eval_Matrix.pckl', 'wb')  # sorting the sorted costs
    f = open(PCKL_STR, 'rb')
    eval_list = pickle.load(f)
    f.close()
    # eval_list has hierarchy [run#][0][sens#][0][test_no][0][iteration][trip#]
    # test no is just one of the exercises in the testing list
    # iteration is one of the 9 times it runs per exercise
    # trip is the (exercise, guessed exercise, time taken)
    runs_for_eval = 0

i = 0
t_init = time.time()
for run_no in range(runs_for_eval):

    # Initialising Lists
    traning_list = []
    testing_list = []
    run_eval_list = []

    # Creating Sets for testing and training

    patex_list = rd.sample(range(NO_PAT), 25)
    for pat in range(NO_PAT):
        for ex in range(NO_EX):
            if pat in patex_list:
                traning_list.append((pat + 1, ex + 1))
            else:
                testing_list.append((pat + 1, ex + 1))

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

if EVAL_NEEDED:
    f = open(PCKL_STR, 'wb')  # sorting the sorted costs
    pickle.dump(eval_list, f)
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

            for i in range(TESTING_NO):
                trip = eval_list[run][0][sens][0][i][0][k_ind]  # reading the triple (execise, guessed exercise,
                # time taken)
                # print(run, sens, i, k_ind)
                t_ave -= trip[2]  # keeps a running total of the time taken
                if trip[0] == trip[1]:
                    acc += 1

        p_acc = 100 * acc / (RUNS * TESTING_NO)
        if K_VAL_ANALYSIS:
            print(legend_labels[sens], ' using a k=', k_ind + 1, ' has accuracy of ', str(p_acc), '%')
        if k_ind + 1 in [1, 3, 5, 7]:
            k_acc.append(p_acc)
            k_val.append(k_ind + 1)

    # Plot results depending on k
    plt.plot(k_val, k_acc, label=legend_labels[sens])

t_ave /= RUNS * 9 * 63 * 4
print('Average Time Taken', t_ave)
plt.title('Accuracy when comparing different Neighbors')
plt.xlabel('No. of nearest neighbors Considered')
plt.ylabel('Accuracy (%)')
plt.xticks(np.arange(1, 9, step=2))
plt = plt
plt.legend(framealpha=0.5)
if UPDATE_FIGS:
    plt.savefig('Accuracy_vs_K.png')

# Sensor Confidence in indexable form
# [act,acw,dc,pm]
# conf_vales = np.array([0.633, 0.4, 0.8, 0.333])
# best_k_val = np.array([1, 1, 1, 1])
conf_vales = np.array([0.6, 0.467, 0.8, 0.367])
best_k_val = np.array([3, 3, 3, 3])
# conf_vales = np.array([0.667, 0.5, 0.767, 0.267])
# best_k_val = np.array([5, 5, 5, 5])
# conf_vales = np.array([0.7, 0.467, 0.833, 0.2])
# best_k_val = np.array([7, 7, 7, 7])
# normalising the confidence numbers
conf_vales = conf_vales / sum(conf_vales)
# lists to hold the accuracy of each exercise
# [act,acw,dc,pm,combined]
accuracies = np.zeros((5, 1))
# initialising confidence lists
conf_guesses = []  # list to hold the confidences of all top guesses
conf_val_corr = []  # list to hold the confidences if the guess is correct
for run in range(RUNS):
    for i in range(TESTING_NO):
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

            # for the combined result using bayesian estimation
            mm_res[est_ex - 1, 0] += conf_vales[sens]  # provides weighting based on confidence

        final_ex = np.argmax(mm_res) + 1
        conf_guesses.append(mm_res.max)  # adding the max value to the list

        if final_ex == actual_ex:
            conf_val_corr.append(mm_res.max)
            accuracies[4, 0] += 1

print()
for i in range(4):
    print(legend_labels[i], ' using a k=', best_k_val[i], ' has accuracy of ',
          accuracies[i, 0] / (RUNS * TESTING_NO * 0.01), '%')
print()

final_acc = accuracies[4, 0] / (RUNS * TESTING_NO * 0.01)

print('The Combined Estimate has accuracy of ', accuracies[4, 0] / (RUNS * TESTING_NO * 0.01), '%')

plt.plot([1, 3, 5, 7], [76.7, 80, 70, 73.3], label='Combined Result')
plt.plot([1, 7], [100 / 8, 100 / 8], label='Random Choice')

plt.legend(loc="right", framealpha=0.7)
plt.savefig('Accuracy_vs_K_and_Final.png')

# Confidence Analysis
print()
print('Average confidence of all guesses = {:5.2f}%'.format(100 * sum(conf_guesses) / len(conf_guesses)))
print('Average confidence of correct guesses = {:5.2f}%'.format(100 * sum(conf_val_corr) / len(conf_val_corr)))

fig = plt.figure(3, figsize=(6, 4))
ax = fig.add_subplot(111)
bp = ax.boxplot([conf_guesses, conf_val_corr])
ax.set_title('Confidence Box-plot')
ax.set_ylabel('Accuracy')
ax.set_xticklabels(['Confidence of all guesses', 'Confidence of Correct Guesses'])
if UPDATE_FIGS:
    fig.savefig('Boxplot.png', bbox_inches='tight')

if EVAL_NEEDED:
    print('That took a total of ', str((time.time() - t_init) / 3600), 'hours')
