import pandas as pd
from sklearn import preprocessing

import DecisionTree
import KNN
import LogisticRegression
import NaiveBayes
import kmeans
import Tools

# df = Dataframe ,X1 = features without ohe , X = features with ohe ,y = labels
df, X1, y = Tools.data_generator()
# one hot encoding
X = Tools.one_hot_encoding(df, X1)
# normalizing data
scaler = preprocessing.MinMaxScaler()
X_scaled_arr = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_arr, columns=X.columns)


# ***************** Models ********************

def models(input):
    if input == '0':
        NaiveBayes.naive_bayes(X,X1, y)
    if input == '1':
        KNN.Knn_main(X_scaled, y)
    if input == '2':
        DecisionTree.decision_tree(X, y)
    if input == '3':
        LogisticRegression.logistic_regression(X, y)
    if input == '4':
        kmeans.k_means(X_scaled)


if __name__ == "__main__":
    while True:
        user_input = input(" \n 0: Naive Bayes \n 1: Knn \n 2: Decision Tree \n 3: Logistic Regression \n 4: K means \n Enter the model number:")
        models(user_input)
