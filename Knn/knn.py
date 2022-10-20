import numpy as np
import pandas as pd
from data_generator import dataGenerator


# difference between a test sample and a train sample
def difference(test_sample, train_sample):
    dif = 0
    for i in range(0, 22):
        if isinstance(train_sample[i], str):
            if train_sample[i] != test_sample[i]:
                dif = dif + 1
        else:
            dif = dif + abs(float(train_sample[i]) - float(test_sample[i]))
    return dif


# choose the k nearest neighbour
def knn(sim_matrix_row, k):
    neighbors = np.argpartition(sim_matrix_row, k)
    return neighbors[0:k]


# predict the label for test sample
def predict(sim_matrix_row, neighbors):
    satisfied = 0
    unsatisfied = 0
    for i in neighbors:
        if train_set[i][len(train_set[0]) - 1] == 'satisfied':
            satisfied += 1 / sim_matrix_row[i]
        else:
            unsatisfied += 1 / sim_matrix_row[i]
    if satisfied > unsatisfied:
        return 'satisfied'
    return 'neutral or dissatisfied'


# the main method
def main():
    error_row = np.zeros(3)
    K = [10, 20, 50]
    for k in range(3):
        sim_matrix = np.zeros([len(test_set), len(train_set)])
        error = 0
        for i in range(len(test_set)):
            for j in range(len(train_set)):
                sim_matrix[i][j] = difference(test_set[i], train_set[j])
        for i in range(len(test_set)):
            if predict(sim_matrix[i], knn(sim_matrix[i], K[k])) != test_set[i][-1]:
                error += 1
        error_row[k] = error / len(test_set)
    return error_row


if __name__ == '__main__':
    # load dataset
    df = pd.read_csv("train.csv")
    df = df.iloc[:300, 2:]
    dg = dataGenerator(df)

    total_error_list = np.zeros([5, 3])

    for i in range(5):
        train_set, test_set = dg.train_and_test()
        error_row_list = main()
        total_error_list[i] = error_row_list
        print(error_row_list)
