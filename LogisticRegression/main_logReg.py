import pandas as pd
import numpy as np
import LogisticRegression
import sys

# Select 0 is MAP
# Select 1 is ML

fp_train = "./bank-note/train.csv"
fp_test = "./bank-note/test.csv"


def processData():
    train_data = pd.read_csv(fp_train, header=None)
    data = train_data.values
    num_col = data.shape[1]
    train_x = np.copy(data)
    train_x[:, num_col - 1] = 1
    train_y = data[:, num_col - 1]
    train_y = 2 * train_y - 1

    test_data = pd.read_csv(fp_test, header=None)
    data = test_data.values
    num_col = data.shape[1]
    test_x = np.copy(data)
    test_x[:, num_col - 1] = 1
    test_y = data[:, num_col - 1]
    test_y = 2 * test_y - 1

    return test_x, test_y, train_x, train_y


def buildLR(train_x, train_y, test_x, test_y, select):
    model = LogisticRegression.LogisticRegression()
    variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

    for var in variances:
        model.set_v(var)
        print("var:", var)
        if select == 0:
            print("MAP")
            w = model.train_MAP(train_x, train_y)
        else:
            print("ML")
            w = model.train_ML(train_x, train_y)

        pred = np.matmul(train_x, w)
        pred[pred > 0] = 1
        pred[pred <= 0] = -1
        train_err = (
            np.sum(np.abs(pred - np.reshape(train_y, (-1, 1)))) / 2 / train_y.shape[0]
        )

        pred = np.matmul(test_x, w)
        pred[pred > 0] = 1
        pred[pred <= 0] = -1

        test_err = (
            np.sum(np.abs(pred - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
        )
        print("train_error: ", train_err, " test_error: ", test_err)


if __name__ == "__main__":
    select = int(sys.argv[1])
    train_x, train_y, test_x, test_y, = processData()
    buildLR(train_x, train_y, test_x, test_y, select)
