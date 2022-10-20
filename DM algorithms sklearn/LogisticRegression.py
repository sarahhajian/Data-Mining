import sys
from io import StringIO
from statistics import mean

import numpy as np

from matplotlib import pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import Tools






def loss_list(model,x_train,y_train):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    model.fit(x_train, y_train)
    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()
    loss_list = []
    for line in loss_history.split('\n'):
        if len(line.split("loss: ")) == 1:
            continue
        loss_list.append(float(line.split("loss: ")[-1]))
    return loss_list


def logistic_regression(X, y):
    # nemoodar loss bar asas learning rate haye motefavet
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0 ,stratify=y)
    alpha_loss_list = []

    for l_rate in [0.0005, 0.1, 0.2, 0.3, 0.5, 6]:
        SGDClf = SGDClassifier(max_iter=100, loss='log_loss', learning_rate='invscaling', eta0=l_rate, verbose=1, )
        lossList = loss_list(SGDClf,x_train,y_train)
        alpha_loss_list.append(lossList)

    # the optimal learning rate

    SGDClf = SGDClassifier(max_iter=100, loss='log_loss', learning_rate='optimal', verbose=1)
    lossList = loss_list(SGDClf,x_train, y_train)
    alpha_loss_list.append(lossList)


    plt.figure()
    for i in range(len(alpha_loss_list)):
        plt.plot(np.arange(len(alpha_loss_list[i])), alpha_loss_list[i])
        plt.xlabel("Iterations");
        plt.ylabel("Loss")
    plt.show()
    # stratified k fold
    testlist_accuracy, trainlist_accuracy =Tools.evaluation(SGDClassifier(loss='log_loss'),StratifiedKFold(n_splits=5),X,y)
    print(" mean test Error: ",1-mean(testlist_accuracy) )