import numpy as np


import dataGenerator as dg
from scipy.stats import norm
import pandas as pd


class TrainModel:

    def __init__(self, dataframe, train_set):
        self.df = dataframe
        self.train_set = train_set

    # calculate probability based on a unique value
    def nominal_probability(self, column, unique_value, number_of_unique_values):
        num = 0
        for i in range(len(self.df.index)):
            if self.df[column].iloc[i] == unique_value:
                num += 1
        length = len(self.df.index)
        probability = (1.0 + num) / (length + number_of_unique_values)
        return probability

    # calculate mean and std of numeric values
    def mean_std(self, column_name):
        mean = self.df[column_name].mean()
        std = self.df[column_name].std()
        if std == 0:
            std = 0.5
        return mean, std

    # return training results
    def train_result(self):
        result = []
        for col in self.df.columns[:-1]:
            if self.df.dtypes[col] == 'object':
                unique_values = self.train_set[col].unique()
                value_dict = dict()
                for value in unique_values:
                    value_dict[value] = self.nominal_probability(col, value, len(unique_values))
                result.append(value_dict)
            else:
                mean, std = self.mean_std(col)
                result.append({
                    'mean': mean,
                    'std': std
                })
        return result

    # probability of label
    def label_probability(self):
        return len(self.df) / len(self.train_set)


def similarity(test_array, probability_list):  #
    num_index = [2,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    nom_index = [0, 1, 3, 4]
    similarity = 1
    for i in nom_index:
        title = probability_list[i]
        choices = list(title)
        for k in range(len(title)):
            if test_array[i] == choices[k]:
                similarity *= title[choices[k]]
    for j in num_index:
        title = probability_list[j]
        mean = title['mean']
        std = title['std']
        pr_num = norm.pdf(test_array[j], loc=mean, scale=std)
        similarity *= pr_num
    return similarity


def main():
    error = 0
    test_list = test.to_numpy()
    for i in range(len(test)):
        if similarity(test_list[i], satisfied_result) * TrainModel(satisfied_df, train).label_probability() >= similarity(
                test_list[i], unsatisfied_result) * TrainModel(unsatisfied_df, train).label_probability():
            if 'satisfied' != test_list[i][-1]:
                error += 1
        else:
            if 'neutral or dissatisfied' != test_list[i][-1]:
                error += 1
    mean_error = error / len(test)

    return mean_error


if __name__ == '__main__':
    # loading data
    df = pd.read_csv("train.csv")
    df = df.iloc[:300, 2:]
    error_list = np.zeros(5)

    for i in range(5):
        train, test = dg.train_and_test(df)
        satisfied_df, unsatisfied_df = dg.split_dataset(train)

        satisfied_result = TrainModel(satisfied_df, train).train_result()
        unsatisfied_result = TrainModel(unsatisfied_df, train).train_result()
        error_list[i] = main()
    print(error_list.mean())





