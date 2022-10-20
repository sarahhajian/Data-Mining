# Data generator methods
def train_and_test(df):
    training_data = df.sample(frac=0.8)
    testing_data = df.drop(training_data.index)
    return training_data, testing_data


def split_dataset(train_df):
    satisfied_df = train_df[train_df['satisfaction'] == 'satisfied']
    unsatisfied_df = train_df[train_df['satisfaction'] == 'neutral or dissatisfied']
    return satisfied_df, unsatisfied_df
