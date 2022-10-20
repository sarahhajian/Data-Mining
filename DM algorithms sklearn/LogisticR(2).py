import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import Tools

df, X1, y = Tools.data_generator()
# one hot encoding
X = Tools.one_hot_encoding(df, X1)


def to_numeric(list):
    for i in range(len(list)):
        if list[i] == 'neutral or dissatisfied':
            list[i] = 0
        else:
            list[i] = 1
    return np.array(list, dtype=float)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
iterations = [50,60,70,80,90,100,110,120,130,140]
loss_list = []
for i in range(len(iterations)):
    SGDClf = SGDClassifier(max_iter=iterations[i], loss='log_loss')
    SGDClf.fit(x_train, y_train)
    y_pred_no = SGDClf.predict(x_test)

    y_true = to_numeric(y_test.values.tolist())
    y_pre = to_numeric(y_pred_no)

    lo = log_loss(y_true, y_pre)
    loss_list.append(lo)
print(loss_list)


plt.figure()

plt.plot(iterations, loss_list)
plt.xlabel(" max iterations");
plt.ylabel("Loss")
plt.show()

