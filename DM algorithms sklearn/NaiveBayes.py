import pandas as pd

from numpy import mean
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.naive_bayes import GaussianNB

import Tools


def probability_table(df):
    result = dict()
    for colname in df.iloc[:, :22]:
        if df[colname].dtypes != 'int64' and df[colname].dtypes != 'float':
            feature_name = df.groupby(colname).size().axes
            list = df.groupby(colname).size().values
            tr = dict()
            result[colname] = tr
            for i in range(len(list)):
                tr[feature_name[0].array[i]] = list[i] / len(df)

        else:
            result[colname] = {
                'mean': df[colname].mean(),
                'std': df[colname].std()
            }
    return result


def naive_bayes(X,X1, y):
    kfold = KFold(n_splits=5)
    s_kfold = StratifiedKFold(n_splits=5)

    error_list = []
    for eva_method in [kfold, s_kfold]:
        error_var_smoothing = []
        for var_smoothing in [1e-9, 10e-9, 100e-9]:
            G_NaiveBayes = GaussianNB(var_smoothing=var_smoothing)
            testlist_accuracy, trainlist_accuracy = Tools.evaluation(G_NaiveBayes, eva_method, X, y)
            error_var_smoothing.append(1 - mean(testlist_accuracy))
        error_list.append(error_var_smoothing)
    #  Error Dataframe
    list = [[1e-9, 10e-9, 100e-9], error_list[0], error_list[1], ]
    df_error = pd.DataFrame(list, columns=['var_smoothing', 'test error(k-fold)', 'test error(stratified k-fold)'],
                            dtype=float)
    print(df_error)

    df_train = Tools.train_fold( kfold, X1, y)
    satisfied_df = df_train[df_train['satisfaction'] == 'satisfied']
    unsatisfied_df = df_train[df_train['satisfaction'] == 'neutral or dissatisfied']

    print("\n******* satisfied *******\n")
    result_sat = probability_table(satisfied_df)
    result_unsat = probability_table(unsatisfied_df)
    for key in result_sat.keys():
        print(key, ": ", result_sat[key])
    print("\n******* unsatisfied *******\n")
    for key in result_unsat.keys():
        print(key, ": ", result_unsat[key])
