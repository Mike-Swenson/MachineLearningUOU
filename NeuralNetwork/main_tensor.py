import pandas as pd
import numpy as np
import Tensflow_FFNN
import sys


fp = "./bank-note/train.csv"
fp_Test = "./bank-note/test.csv"


def processData():
    train_data = pd.read_csv(fp, header=None)
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
    input_dim = train_x.shape[1]
    output_dim = 1

    width_list = [5, 10, 25, 50, 100]
    n_layers = [3, 5, 9]
    for n_layer in n_layers:
        for width in width_list:
            print("layer: ", n_layer, "width: ", width)
            s = [input_dim, output_dim]
            for _ in range(n_layer):
                s.insert(1, width)

            model = Tensflow_FFNN.TFNN(s, select)

            model.train(train_x.reshape([-1, input_dim]), train_y.reshape([-1, 1]))
            pred = model.fit(train_x)

            pred[pred > 0] = 1
            pred[pred <= 0] = -1
            train_err = (
                np.sum(np.abs(pred - np.reshape(train_y, (-1, 1))))
                / 2
                / train_y.shape[0]
            )

            pred = model.fit(test_x)
            pred[pred > 0] = 1
            pred[pred <= 0] = -1

            test_err = (
                np.sum(np.abs(pred - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
            )
            print("train_error: ", train_err, " test_error: ", test_err)


if __name__ == "__main__":
    select = int(sys.argv[1])
    train_x, train_y, test_x, test_y, = processData()
    buildNN(train_x, train_y, test_x, test_y, select)
