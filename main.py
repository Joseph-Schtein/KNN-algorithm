import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split


def run_hc(file):
    #path = r'C:\Users\yossi\OneDrive\Documents\machine_learning\Assignment_3\HC_Body_Temperature.txt'
    path = file
    CSV_COLUMN_NAMES = ['BodyTemp', 'Gender', 'Weight']
    df = pd.read_csv(path, sep='   ' or '    ', engine='python', names=CSV_COLUMN_NAMES, header=0)
    df = df.replace({'Gender': {2: 0}})
    train_results_list = np.zeros(shape=(5, 3))
    test_results_list = np.zeros(shape=(5, 3))

    for i in range(500):
        train, test = train_test_split(df, test_size=0.500)
        train_results_list += algorithm(train, train, 'BodyTemp', 'Weight', 'Gender')  # Empirical error
        test_results_list += algorithm(train, test, 'BodyTemp', 'Weight', 'Gender') # Test error

    for i in range(5):
        for j in range(3):
            train_results_list[i][j] = train_results_list[i][j] / 500
            test_results_list[i][j] = test_results_list[i][j] / 500

    tmp = 1
    print("train error")
    for i in range(5):
        print("for k = ", i + tmp)
        tmp = tmp + 1
        for j in range(3):
            print("for p = ", j + 1, " ", train_results_list[i][j])
        print("\n")

    tmp = 1
    print("test error\n\n")
    for i in range(5):
        print("for k = ", i + tmp)
        tmp = tmp + 1
        for j in range(3):
            print("for p = ", j + 1, " ", test_results_list[i][j])
        print("\n")

    return train_results_list, test_results_list


def algorithm(train, test, x, y, label):
    train_values_x = list(train[x])
    train_values_y = list(train[y])
    test_values_x = list(test[x])
    test_values_y = list(test[y])

    train_labels = list(train[label])
    test_labels = list(test[label])
    test_prediction_p1 = np.zeros(shape=65)
    test_prediction_p2 = np.zeros(shape=65)
    test_prediction_infinity = np.zeros(shape=65)

    output = np.zeros(shape=(5, 3))
    index = 0
    for k in [1, 3, 5, 7, 9]:
        manh_dis = np.zeros(shape=k)
        euc_dis = np.zeros(shape=k)
        infinity_dis = np.zeros(shape=k)

        value_closer_p1 = np.zeros(shape=(k, 3))
        value_closer_p2 = np.zeros(shape=(k, 3))
        value_closer_infinity = np.zeros(shape=(k, 3))

        for i in range(64):  # Passing on all the test data for the algorithm

            dot_x = test_values_x[i]
            dot_y = test_values_y[i]
            initialize = True

            for j in range(64):  # Comput the distance according to l1, l2 and l-infinity over all the train values
                tmp_distance_for_p1 = abs(dot_x - train_values_x[j]) + abs(dot_y - train_values_y[j])
                tmp_distance_for_p2 = math.sqrt(
                    math.pow(dot_x - train_values_x[j], 2) + math.pow(dot_y - train_values_y[j], 2))
                tmp_distance_for_infinity = max(abs(dot_x - train_values_x[j]), abs(dot_y - train_values_y[j]))

                tmp_distance_for_p1 = int(tmp_distance_for_p1 * 10000) / 10000
                tmp_distance_for_p2 = int(tmp_distance_for_p2 * 10000) / 10000
                tmp_distance_for_infinity = int(tmp_distance_for_infinity * 10000) / 10000

                if not initialize:
                    r = k - 1
                    if r != 0:
                        while r >= 0 and manh_dis[r] > tmp_distance_for_p1:
                            if r != k - 1:
                                manh_dis[r + 1] = manh_dis[r]
                                value_closer_p1[r + 1] = value_closer_p1[r]
                            r = r - 1

                        if r != k - 1:
                            manh_dis[r + 1] = tmp_distance_for_p1
                            value_closer_p1[r + 1][0] = train_values_x[j]
                            value_closer_p1[r + 1][1] = train_values_y[j]
                            value_closer_p1[r + 1][2] = train_labels[j]

                    elif manh_dis[0] > tmp_distance_for_p1 and k == 1:
                        manh_dis[0] = tmp_distance_for_p1
                        value_closer_p1[0][0] = train_values_x[j]
                        value_closer_p1[0][1] = train_values_y[j]
                        value_closer_p1[0][2] = train_labels[j]

                else:
                    if k == 1 or j == 0:
                        manh_dis[j] = tmp_distance_for_p1
                        value_closer_p1[j][0] = train_values_x[j]
                        value_closer_p1[j][1] = train_values_y[j]
                        value_closer_p1[j][2] = train_labels[j]

                    else:
                        t = k - 1

                        while t >= 0 and (manh_dis[t] > tmp_distance_for_p1 or manh_dis[t] == 0):
                            if manh_dis[t] != 0 and t != k - 1:
                                manh_dis[t + 1] = manh_dis[t]
                                value_closer_p1[t + 1] = value_closer_p1[t]
                            t = t - 1

                        if t != k - 1:
                            manh_dis[t + 1] = tmp_distance_for_p1
                            value_closer_p1[t + 1][0] = train_values_x[j]
                            value_closer_p1[t + 1][1] = train_values_y[j]
                            value_closer_p1[t + 1][2] = train_labels[j]

                if not initialize:
                    r = k - 1
                    if r != 0:
                        while r >= 0 and euc_dis[r] > tmp_distance_for_p2:
                            if r != k - 1:
                                euc_dis[r + 1] = euc_dis[r]
                                value_closer_p2[r + 1] = value_closer_p2[r]
                            r = r - 1

                        if r != k - 1:
                            euc_dis[r + 1] = tmp_distance_for_p2
                            value_closer_p2[r + 1][0] = train_values_x[j]
                            value_closer_p2[r + 1][1] = train_values_y[j]
                            value_closer_p2[r + 1][2] = train_labels[j]

                    elif euc_dis[0] > tmp_distance_for_p2:
                        euc_dis[0] = tmp_distance_for_p2
                        value_closer_p2[0][0] = train_values_x[j]
                        value_closer_p2[0][1] = train_values_y[j]
                        value_closer_p2[0][2] = train_labels[j]

                else:
                    if k == 1 or j == 0:
                        euc_dis[j] = tmp_distance_for_p2
                        value_closer_p2[j][0] = train_values_x[j]
                        value_closer_p2[j][1] = train_values_y[j]
                        value_closer_p2[j][2] = train_labels[j]

                    else:
                        t = k - 1

                        while t >= 0 and (euc_dis[t] > tmp_distance_for_p2 or euc_dis[t] == 0):
                            if euc_dis[t] != 0 and t != k - 1:
                                euc_dis[t + 1] = euc_dis[t]
                                value_closer_p2[t + 1] = value_closer_p2[t]
                            t = t - 1

                        if t != k - 1:
                            euc_dis[t + 1] = tmp_distance_for_p2
                            value_closer_p2[t + 1][0] = train_values_x[j]
                            value_closer_p2[t + 1][1] = train_values_y[j]
                            value_closer_p2[t + 1][2] = train_labels[j]

                if not initialize:
                    r = k - 1
                    if r != 0:
                        while r >= 0 and infinity_dis[r] > tmp_distance_for_infinity:
                            if r != k - 1:
                                infinity_dis[r + 1] = infinity_dis[r]
                                value_closer_infinity[r + 1] = value_closer_infinity[r]
                            r = r - 1

                        if r != k - 1:
                            infinity_dis[r + 1] = tmp_distance_for_infinity
                            value_closer_infinity[r + 1][0] = train_values_x[j]
                            value_closer_infinity[r + 1][1] = train_values_y[j]
                            value_closer_infinity[r + 1][2] = train_labels[j]

                    elif infinity_dis[0] > tmp_distance_for_infinity:
                        infinity_dis[0] = tmp_distance_for_infinity
                        value_closer_infinity[0][0] = train_values_x[j]
                        value_closer_infinity[0][1] = train_values_y[j]
                        value_closer_infinity[0][2] = train_labels[j]

                else:
                    if k == 1 or j == 0:
                        infinity_dis[j] = tmp_distance_for_infinity
                        value_closer_infinity[j][0] = train_values_x[j]
                        value_closer_infinity[j][1] = train_values_y[j]
                        value_closer_infinity[j][2] = train_labels[j]

                    else:
                        t = k - 1

                        while t >= 0 and (infinity_dis[t] > tmp_distance_for_infinity or infinity_dis[t] == 0):
                            if infinity_dis[t] != 0 and t != k - 1:
                                infinity_dis[t + 1] = infinity_dis[t]
                                value_closer_infinity[t + 1] = value_closer_infinity[t]
                            t = t - 1

                        if t != k - 1:
                            infinity_dis[t + 1] = tmp_distance_for_infinity
                            value_closer_infinity[t + 1][0] = train_values_x[j]
                            value_closer_infinity[t + 1][1] = train_values_y[j]
                            value_closer_infinity[t + 1][2] = train_labels[j]

                if k == j + 1:
                    initialize = False

            zeros_p1 = 0
            ones_p1 = 0
            zeros_p2 = 0
            ones_p2 = 0
            zeros_infinity = 0
            ones_infinity = 0

            for j in range(k):  # Label the dot

                if value_closer_p1[j][2] == 0:
                    zeros_p1 = zeros_p1 + 1

                else:
                    ones_p1 = ones_p1 + 1

                if value_closer_p2[j][2] == 0:
                    zeros_p2 = zeros_p2 + 1

                else:
                    ones_p2 = ones_p2 + 1

                if value_closer_infinity[j][2] == 0:
                    zeros_infinity = zeros_infinity + 1

                else:
                    ones_infinity = ones_infinity + 1

            if zeros_p1 > ones_p1:
                test_prediction_p1[i] = 0

            else:
                test_prediction_p1[i] = 1

            if zeros_p2 > ones_p2:
                test_prediction_p2[i] = 0

            else:
                test_prediction_p2[i] = 1

            if zeros_infinity > ones_infinity:
                test_prediction_infinity[i] = 0

            else:
                test_prediction_infinity[i] = 1

        mistake_p1 = 0
        mistake_p2 = 0
        mistake_infinity = 0

        for i in range(64):

            if test_labels[i] != test_prediction_p1[i]:
                mistake_p1 = mistake_p1 + 1

            if test_labels[i] != test_prediction_p2[i]:
                mistake_p2 = mistake_p2 + 1

            if test_labels[i] != test_prediction_infinity[i]:
                mistake_infinity = mistake_infinity + 1

        output[index][0] = mistake_p1 / 65
        output[index][1] = mistake_p2 / 65
        output[index][2] = mistake_infinity / 65
        index = index + 1

    return output


def main():
    path = input("insert please the path: " )
    results = run_hc(path)


if __name__ == '__main__':
    main()
