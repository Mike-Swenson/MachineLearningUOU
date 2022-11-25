import pandas as pd
import numpy as np
import SVM
import sys


def processData():
    train_data = pd.read_csv("./bank-note/train.csv", header=None)
    raw = train_data.values
    num_col = raw.shape[1]
    num_row = raw.shape[0]
    train_x = np.copy(raw)
    train_x[:, num_col - 1] = 1
    train_y = raw[:, num_col - 1]
    train_y = 2 * train_y - 1

    test_data = pd.read_csv("./bank-note/test.csv", header=None)
    raw = test_data.values
    num_col = raw.shape[1]
    num_row = raw.shape[0]
    test_x = np.copy(raw)
    test_x[:, num_col - 1] = 1
    test_y = raw[:, num_col - 1]
    test_y = 2 * test_y - 1

    return train_x, train_y, test_x, test_y, num_col


def svmReporter(train_x, train_y, test_x, test_y, num_col, select):
    C_set = np.array([100 / 873, 500 / 873, 700 / 873])
    gammas = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 100])

    svm = SVM.SVM()
    # Q1 a
    if select == 1:
        for C in C_set:
            print("C: ", C)
            svm.set_C(C)
            w = svm.train_primal_one(train_x, train_y)
            w = np.reshape(w, (5, 1))

            pred = np.matmul(train_x, w)
            pred[pred > 0] = 1
            pred[pred <= 0] = -1

            train_err = (
                np.sum(np.abs(pred - np.reshape(train_y, (-1, 1))))
                / 2
                / train_y.shape[0]
            )

            pred = np.matmul(test_x, w)
            pred[pred > 0] = 1
            pred[pred <= 0] = -1

            test_err = (
                np.sum(np.abs(pred - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
            )
            print("Q2 AAAAAAAAAAAAAAAAAAAAA")
            print(
                "linear SVM Primal train_error: ", train_err, " test_error: ", test_err
            )
            print("w1: ", w)
            print()
            w = np.reshape(w, (1, -1))

    # Q1 b
    if select == 2:
        for C in C_set:
            print("C: ", C)
            svm.set_C(C)
            w = svm.train_primal_two(train_x, train_y)
            w = np.reshape(w, (5, 1))

            pred = np.matmul(train_x, w)
            pred[pred > 0] = 1
            pred[pred <= 0] = -1

            train_err = (
                np.sum(np.abs(pred - np.reshape(train_y, (-1, 1))))
                / 2
                / train_y.shape[0]
            )

            pred = np.matmul(test_x, w)
            pred[pred > 0] = 1
            pred[pred <= 0] = -1

            test_err = (
                np.sum(np.abs(pred - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
            )
            print("Q2 BBBBBBBBBBBBBBBBBBBBB")
            print(
                "linear SVM Primal train_error: ", train_err, " test_error: ", test_err
            )
            w = np.reshape(w, (1, -1))

            print("w1: ", w)
            print()

    # dual form q2a
    if select == 3:
        for C in C_set:
            w = svm.train_d(train_x[:, [x for x in range(num_col - 1)]], train_y)
            print("Q3 Dual Form")
            print("C:", C)
            print("w2: ", w)

            w = np.reshape(w, (5, 1))

            pred = np.matmul(train_x, w)
            pred[pred > 0] = 1
            pred[pred <= 0] = -1
            train_err = (
                np.sum(np.abs(pred - np.reshape(train_y, (-1, 1))))
                / 2
                / train_y.shape[0]
            )

            pred = np.matmul(test_x, w)
            pred[pred > 0] = 1
            pred[pred <= 0] = -1

            test_err = (
                np.sum(np.abs(pred - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
            )
            print("linear SVM Dual train_error: ", train_err, " test_error: ", test_err)

    # gaussian kernel
    if select == 4:
        c = 0
        for C in C_set:
            for gamma in gammas:
                print(C)
                print("gamma: ", gamma)
                svm.set_gamma(gamma)
                alpha = svm.train_gaussian_kernel(
                    train_x[:, [x for x in range(num_col - 1)]], train_y
                )
                supportV = np.where(alpha > 0)[0]
                # print("Alpha values non zero", supportV)
                print("#sv: ", len(supportV))
                # train
                y = svm.predict_gaussian_kernel(
                    alpha,
                    train_x[:, [x for x in range(num_col - 1)]],
                    train_y,
                    train_x[:, [x for x in range(num_col - 1)]],
                )
                train_err = (
                    np.sum(np.abs(y - np.reshape(train_y, (-1, 1))))
                    / 2.0000
                    / train_y.shape[0]
                )

                # test
                y = svm.predict_gaussian_kernel(
                    alpha,
                    train_x[:, [x for x in range(num_col - 1)]],
                    train_y,
                    test_x[:, [x for x in range(num_col - 1)]],
                )
                test_err = (
                    np.sum(np.abs(y - np.reshape(test_y, (-1, 1))))
                    / 2.0000
                    / test_y.shape[0]
                )
                print(
                    "nonlinear SVM train_error: ", train_err, " test_error: ", test_err,
                )
                old_supp = supportV
                if c > 0:
                    intersect = len(np.intersect1d(supportV, old_supp))
                    print("#intersect: ", intersect)
                c = c + 1


if __name__ == "__main__":
    svmSelect = int(sys.argv[1])
    trainInput, trainLabels, testInput, testLabels, num_col = processData()
    svmReporter(trainInput, trainLabels, testInput, testLabels, num_col, svmSelect)
