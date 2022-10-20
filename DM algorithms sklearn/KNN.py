from statistics import mean

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier

import Tools


def Knn_main(X, y):
    # KNN
    # Stratified k fold
    s_kfold = StratifiedKFold(n_splits=5)
    # k fold
    kfold = KFold(n_splits=5)
    error_method_test = []
    error_method_train = []
    for e_method in [s_kfold, kfold]:
        K = [10, 20, 30, 40, 50]
        error_k_test = []
        error_k_train = []
        for i in range(len(K)):
            error_p_test = []
            error_p_train = []
            for p in range(1, 4):
                knn_clf = KNeighborsClassifier(n_neighbors=K[i], p=p)
                testlist_accuracy, trainlist_accuracy = Tools.evaluation(knn_clf, e_method, X, y)
                error_p_test.append(1 - mean(testlist_accuracy))
                error_p_train.append(1 - mean(trainlist_accuracy))

            error_k_test.append(error_p_test)
            error_k_train.append(error_p_train)
        error_method_test.append(error_k_test)
        error_method_train.append(error_k_train)

    print("Stratified 5 Fold Error : ", '\n', "train : ", error_method_train[0], '\n', "test : ", error_method_test[0],
          '\n')
    print(" 5 Fold Error: ", '\n', "train : ", error_method_train[1], '\n', "test : ", error_method_test[1], '\n')
    # mean for every K
    manhatan_error_train = []
    for list in error_method_train[1]:
        manhatan_error_train.append(list[0])
    manhatan_error_test = []
    for list in error_method_test[1]:
        manhatan_error_test.append(list[0])

    # Plot for k fold cross validation

    plt.plot(K, manhatan_error_train, color='r', label='train error')
    plt.plot(K, manhatan_error_test, color='g', label='test error')
    plt.xlabel("K");
    plt.ylabel("Error ")
    plt.show()

