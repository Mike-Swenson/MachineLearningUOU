#%%
import pandas as pd
import numpy as np
import FFNNq2abcd
import sys

# Select 0 is random weights
# Select 1 is initialize 0

fp_Training = "./bank-note/train.csv"
fp_Test = "./bank-note/test.csv"


def processData():

    train_data = pd.read_csv(fp_Training, header=None)
    # process data
    data = train_data.values
    num_col = data.shape[1]
    train_x = np.copy(data)
    train_x[:, num_col - 1] = 1
    train_y = data[:, num_col - 1]
    train_y = 2 * train_y - 1

    test_data = pd.read_csv(fp_Test, header=None)
    data = test_data.values
    num_col = data.shape[1]
    test_x = np.copy(data)
    test_x[:, num_col - 1] = 1
    test_y = data[:, num_col - 1]
    test_y = 2 * test_y - 1

    return train_x, train_y, test_x, test_y


def buildNN(train_x, train_y, test_x, test_y, select):
    in_d = train_x.shape[1]
    out_d = 1

    width_list = [5, 10, 25, 50, 100]

    # Vary the widths
    for width in width_list:
        s = [in_d, width, width, out_d]
        model = FFNNq2abcd.NN(s)

        model.train(train_x.reshape([-1, in_d]), train_y.reshape([-1, 1]))
        pred = model.fit(train_x)

        pred[pred > 0] = 1
        pred[pred <= 0] = -1
        train_err = (
            np.sum(np.abs(pred - np.reshape(train_y, (-1, 1)))) / 2.0 / train_y.shape[0]
        )

        pred = model.fit(test_x)
        pred[pred > 0] = 1
        pred[pred <= 0] = -1

        test_err = (
            np.sum(np.abs(pred - np.reshape(test_y, (-1, 1)))) / 2.0 / test_y.shape[0]
        )
        print(width)
        print("train_error: ", train_err)
        print("test_error: ", test_err)


if __name__ == "__main__":
    select = int(sys.argv[1])
    train_x, train_y, test_x, test_y, = processData()
    buildNN(train_x, train_y, test_x, test_y, select)
