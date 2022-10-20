import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def data_generator():
    # read dataframe from csv file
    df = pd.read_csv('train.csv', index_col=0)
    # ****** Normalize Data *********
    # clean the dataset
    df = df.drop(["id"], axis='columns')
    df = df.iloc[:400]

    # X = features column (type:df) , y = labels column (type:df)
    X = df.iloc[:, :21]
    y = df['satisfaction']

    return df, X, y


def one_hot_encoding(df, X):
    nominal_col = [col_name for col_name in X.columns if df[col_name].dtypes != 'int64']
    ohe = OneHotEncoder()
    for col_name in nominal_col:
        transformed = ohe.fit_transform(X[[col_name]])
        X[ohe.categories_[0]] = transformed.toarray()
        X = X.drop(col_name, axis='columns')
    return X


def evaluation(model, eva_method, X, y):
    testlist_accuracy = []
    trainlist_accuracy = []
    for train_index, test_index in eva_method.split(X, y):
        x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model.fit(x_train_fold, y_train_fold)
        testlist_accuracy.append(model.score(x_test_fold, y_test_fold))
        trainlist_accuracy.append(model.score(x_train_fold, y_train_fold))
    return testlist_accuracy, trainlist_accuracy

def train_fold(eva_method,X, y):
    for train_index, test_index in eva_method.split(X, y):
        x_train_fold = X.iloc[train_index]
        y_train_fold= y[train_index]
        return  pd.concat([x_train_fold, y_train_fold], join = 'outer', axis = 1)
