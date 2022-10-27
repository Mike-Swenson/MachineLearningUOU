#%%
import pandas as pd
import numpy as np
import LMS
import sys

method = sys.argv[1]


train_data = pd.read_csv("./concrete/train.csv", header=None)
# process data
data = train_data.values
columns = data.shape[1]
rows = data.shape[0]

train_x = np.copy(data)
train_x[:, columns - 1] = 1
train_y = data[:, columns - 1]
train_y = 2 * train_y - 1

test_data = pd.read_csv("./concrete/test.csv", header=None)
data = test_data.values

columns = data.shape[1]
rows = data.shape[0]
test_x = np.copy(data)
test_x[:, columns - 1] = 1
test_y = data[:, columns - 1]
test_y = 2 * test_y - 1

# lms model
lms = LMS.LMS()

if int(method) == 0:
    w = lms.optimize(train_x, train_y)
    print("GD w: ", w)
    tmp = np.reshape(np.squeeze(np.matmul(test_x, w)) - test_y, (-1, 1))
    functionVal = 0.5 * np.sum(np.square(tmp))
    print("GD test_functionVal: ", functionVal)

elif int(method) == 1:
    # SGD
    lms.set_method(1)
    w = lms.optimize(train_x, train_y)
    print("GD w: ", w)
    tmp = np.reshape(np.squeeze(np.matmul(test_x, w)) - test_y, (-1, 1))
    functionVal = 0.5 * np.sum(np.square(tmp))
    print("GD test_functionVal: ", functionVal)

elif int(method) == 2:
    # normal equation
    lms.set_method(2)
    w = lms.optimize(train_x, train_y)
    print("GD w: ", w)
    tmp = np.reshape(np.squeeze(np.matmul(test_x, w)) - test_y, (-1, 1))
    functionVal = 0.5 * np.sum(np.square(tmp))
    print("LA Solve: ", functionVal)
