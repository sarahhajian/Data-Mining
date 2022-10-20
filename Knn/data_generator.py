
import pandas as pd
import numpy as np


class dataGenerator:

    def __init__(self, dataframe):
        self.testing_data = None
        self.training_data = None
        self.df = dataframe

    # randomly split dataset to train and test and convert it to numpy array
    def split_dataset(self):
        training_data = self.df.sample(frac=0.8 )
        testing_data = self.df.drop(training_data.index)

        self.training_data = training_data.to_numpy()
        self.testing_data = testing_data.to_numpy()

        return self.training_data, self.testing_data

    # return normalize train and test sets
    def train_and_test(self):
        index = [self.df.columns.get_loc(col_name) for col_name in self.df.columns if self.df[col_name].dtype != 'object']
        for i in index:
            self.normalize(i)
        self.df = pd.DataFrame(self.df)
        return self.split_dataset()

    # normalize dataset
    def normalize(self, i):
        column = self.df.iloc[:, i]
        min = column.min()
        max = column.max()
        self.df.iloc[:, i] -= min
        domain = max - min
        domain = float(domain)
        self.df.iloc[:, i] /= domain