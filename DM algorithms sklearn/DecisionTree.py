from statistics import mean

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import KFold

import Tools


def decision_tree(X, y):
    kfold = KFold(n_splits=5)
    # plot bezaye maghadir default
    plt.figure(figsize=(50, 50))
    DTree = tree.DecisionTreeClassifier()
    testlist_accuracy, trainlist_accuracy = Tools.evaluation(DTree, kfold, X, y)
    tree.plot_tree(DTree, feature_names=X.columns.tolist(), filled=True)
    plt.show()


    error_list_test=[]
    error_list_train = []
    for min_leaf in range(1,14):
        DTree = tree.DecisionTreeClassifier(min_samples_leaf=min_leaf)
        testlist_accuracy, trainlist_accuracy = Tools.evaluation(DTree, kfold, X, y)
        error_list_test.append(1- mean(testlist_accuracy))
        error_list_train.append(1- mean(trainlist_accuracy))
    # plot bezaye min leaf motefavet
    plt.plot(range(1,14),error_list_test,color='r', label='train error')
    plt.plot(range(1,14), error_list_train,color='g', label='test error')
    plt.xlabel("min sample leaf");
    plt.ylabel("Error")
    plt.show()
    # plot bezaye max depth motefavet
    error_list_test=[]
    error_list_train = []
    for max_depth in range(1,14):
        DTree = tree.DecisionTreeClassifier(max_depth=max_depth)
        testlist_accuracy, trainlist_accuracy = Tools.evaluation(DTree, kfold, X, y)
        error_list_test.append(1 - mean(testlist_accuracy))
        error_list_train.append(1 - mean(trainlist_accuracy))
    # plot bezaye min leaf motefavet
    plt.plot(range(1,14),error_list_test,color='r', label='train error')
    plt.plot(range(1,14), error_list_train,color='g', label='test error')
    plt.xlabel("max depth");
    plt.ylabel("Error")
    plt.show()

