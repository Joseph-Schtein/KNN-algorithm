import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split


def run_hc():
    path = r'C:\Users\yossi\OneDrive\Documents\machine_learning\Assignment_3\HC_Body_Temperature.txt'
    CSV_COLUMN_NAMES = ['BodyTemp', 'Gender', 'Weight']
    df = pd.read_csv(path, sep='   ' or '    ', engine='python', names=CSV_COLUMN_NAMES, header=0)
    df = df.replace({'Gender': {2: 0}})
    results_list = np.zeros(shape=df.shape[0])

    for i in range(100):
        results_list[i]=algorithm(df, 'BodyTemp', 'Weight', 'Gender')

    return results_list


def algorithm(df, x, y, label):
    train, test = train_test_split(df, test_size=0.500)

    train_values_x = list(train[x])
    train_values_y = list(train[y])
    test_values_x = list(test[x])
    test_values_y = list(test[y])
    train_size = train.shape[0]
    test_size = test.shape[0]

    train_labels = list(train[label])
    test_labels = list(test[label])
    test_prediction_p1 = np.zeros(shape=65)
    test_prediction_p2 = np.zeros(shape=65)
    test_prediction_infinity = np.zeros(shape=65)

    output = np.zero(shape=(5, 3))

    for k in [1, 3, 5, 7, 9]:
        manh_dis = np.zero(shape=k)
        euc_dis = np.zero(shape=k)
        infinity_dis = np.zero(shape=k)

        value_closer_p1 = np.zero(shape=(k, 3))
        value_closer_p2 = np.zero(shape=(k, 3))
        value_closer_infinity = np.zero(shape=(k, 3))

        for i in range(65): # Passing on all the test data for the algorithm
            dot_x = test_values_x[i]
            dot_y = test_values_y[i]
            label = test_labels[i]
            initialize = True
            for j in range(65): # Comput the distance according to l1, l2 and l-infinity over all the train values
                tmp_distance_for_p1 = abs(dot_x - train_values_x[j]) + abs(dot_y - train_values_y[j])
                tmp_distance_for_p2 = math.sqrt(math.pow(dot_x - train_values_x[j], 2) + math.pow(dot_y - train_values_y[j], 2))
                tmp_distance_for_infinity = max(abs(dot_x - train_values_x[j]), abs(dot_y - train_values_y[j]))

                tmp_dot = np.zero(shape=3)

                if not initialize:
                    for r in range(k):
                        if manh_dis[r] > tmp_distance_for_p1:
                            tmp_variable = manh_dis[r]
                            manh_dis[r] = tmp_distance_for_p1
                            tmp_distance_for_p1 = tmp_variable

                            tmp_dot = value_closer_p1[r]
                            value_closer_p1[r][0] = dot_x
                            value_closer_p1[r][1] = dot_y
                            value_closer_p1[r][2] = label
                            if r+1 < k:
                                value_closer_p1[r+1] = tmp_dot

                else:
                    manh_dis[j] = tmp_distance_for_p1
                    value_closer_p1[j][0] = dot_x
                    value_closer_p1[j][1] = dot_y
                    value_closer_p1[j][2] = label

                tmp_dot = np.zero(shape=3)

                if not initialize:
                    for r in range(k):
                        if euc_dis[r] > tmp_distance_for_p2:
                            tmp_variable = euc_dis[r]
                            euc_dis[r] = tmp_distance_for_p2
                            tmp_distance_for_p2 = tmp_variable

                            tmp_dot = value_closer_p2[r]
                            value_closer_p2[r][0] = dot_x
                            value_closer_p2[r][1] = dot_y
                            value_closer_p2[r][2] = label
                            if r + 1 < k:
                                value_closer_p2[r + 1] = tmp_dot

                else:
                    euc_dis[j] = tmp_distance_for_p2
                    value_closer_p2[j][0] = dot_x
                    value_closer_p2[j][1] = dot_y
                    value_closer_p2[j][2] = label

                tmp_dot = np.zero(shape=3)

                if not initialize:
                    for r in range(k):
                        if infinity_dis[r] > tmp_distance_for_infinity:
                            tmp_variable = infinity_dis[r]
                            infinity_dis[r] = tmp_distance_for_infinity
                            tmp_distance_for_infinity = tmp_variable

                            tmp_dot = value_closer_infinity[r]
                            value_closer_infinity[r][0] = dot_x
                            value_closer_infinity[r][1] = dot_y
                            value_closer_infinity[r][2] = label
                            if r + 1 < k:
                                value_closer_infinity[r + 1] = tmp_dot

                else:
                    infinity_dis[j] = tmp_distance_for_infinity
                    value_closer_infinity[j][0] = dot_x
                    value_closer_infinity[j][1] = dot_y
                    value_closer_infinity[j][2] = label

                if k == j:
                    initialize = False

            for j in range(k):  # Label the dot
                zeros_p1 = 0
                ones_p1 = 0
                zeros_p2 = 0
                ones_p2 = 0
                zeros_infinity = 0
                ones_infinity = 0

                if value_closer_p1[j][2] == 0:
                    zeros_p1 = zeros_p1+1

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



    return output


def main():
    run_hc()


if __name__ == '__main__':
    main()
