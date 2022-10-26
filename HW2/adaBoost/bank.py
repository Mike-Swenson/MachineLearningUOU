import pandas as pd
import weightedDT as dt
import math
import numpy as np
import matplotlib.pyplot as plt
import random

import sys

iteration = sys.argv[1] if sys.argv[1] else 1
for i in range(int(iteration)):
    columns = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y",
    ]
    types = {
        "age": int,
        "job": str,
        "marital": str,
        "education": str,
        "default": str,
        "balance": int,
        "housing": str,
        "loan": str,
        "contact": str,
        "day": int,
        "month": str,
        "duration": int,
        "campaign": int,
        "pdays": int,
        "previous": int,
        "poutcome": str,
        "y": str,
    }
    train_data = pd.read_csv("./bank/train.csv", names=columns, dtype=types)
    train_size = len(train_data.index)

    # convert numeric to binary
    numeric_features = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]
    for c in numeric_features:
        median = train_data[c].median()
        train_data[c] = train_data[c].apply(lambda x: 0 if x < median else 1)

    test_data = pd.read_csv("./bank/test.csv", names=columns, dtype=types)
    test_size = len(test_data.index)
    for c in numeric_features:
        median = test_data[c].median()
        test_data[c] = test_data[c].apply(lambda x: 0 if x < median else 1)

    # set features and label
    features = {
        "age": [0, 1],
        "job": [
            "admin.",
            "unknown",
            "unemployed",
            "management",
            "housemaid",
            "entrepreneur",
            "student",
            "blue-collar",
            "self-employed",
            "retired",
            "technician",
            "services",
        ],
        "marital": ["married", "divorced", "single"],
        "education": ["unknown", "secondary", "primary", "tertiary"],
        "default": ["yes", "no"],
        "balance": [0, 1],
        "housing": ["yes", "no"],
        "loan": ["yes", "no"],
        "contact": ["unknown", "telephone", "cellular"],
        "day": [0, 1],
        "month": [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
        "duration": [0, 1],
        "campaign": [0, 1],
        "pdays": [0, 1],
        "previous": [0, 1],
        "poutcome": ["unknown", "other", "failure", "success"],
    }
    label = {"y": ["yes", "no"]}

    T = random.randrange(1, 500)
    train_size = len(train_data.index)
    test_size = len(test_data.index)

    alphas = [0 for x in range(T)]

    weights = np.array([1 / train_size for x in range(train_size)])

    trainingError = [0 for x in range(T)]
    testingError = [0 for x in range(T)]

    train_errT = [0 for x in range(T)]
    test_errT = [0 for x in range(T)]

    train_r = [0 for x in range(T)]
    test_r = [0 for x in range(T)]

    train_py = np.array([0 for x in range(train_size)])
    test_py = np.array([0 for x in range(test_size)])

    print("Print total iterations: ", T)
    for t in range(T):

        print("Iteration: ", t)
        # ID3 stumps
        # Ada Boost Max depth 1
        dt_generator = dt.weightedID3(feature_selection=0, max_depth=1)
        decision_tree = dt_generator.makeTree(train_data, features, label, weights)

        # error
        train_data["py"] = dt_generator.predict(decision_tree, train_data)
        tmp = train_data.apply(lambda row: 1 if row["y"] == row["py"] else 0, axis=1)
        err = 1 - tmp.sum() / train_size
        trainingError[t] = err

        # alpha
        tmp = train_data.apply(lambda row: 1 if row["y"] == row["py"] else -1, axis=1)
        tmp = np.array(tmp.tolist())
        w = weights[tmp == -1]
        err = np.sum(w)
        alpha = 0.5 * math.log((1 - err) / err)
        alphas[t] = alpha

        # Caclulate weights
        weights = np.exp(tmp * -alpha) * weights
        total = np.sum(weights)
        weights = weights / total

        # testing error
        test_data["py"] = dt_generator.predict(decision_tree, test_data)
        tmp = test_data.apply(lambda row: 1 if row["y"] == row["py"] else 0, axis=1)
        testingError[t] = 1 - tmp.sum() / test_size

        # combined prediction so far
        # train
        py = np.array(train_data["py"].tolist())
        py[py == "yes"] = 1
        py[py == "no"] = -1
        py = py.astype(int)
        train_py = train_py + py * alpha
        py = py.astype(str)
        py[train_py > 0] = "yes"
        py[train_py <= 0] = "no"
        train_data["py"] = pd.Series(py)
        tmp = train_data.apply(lambda row: 1 if row["y"] == row["py"] else 0, axis=1)
        err = 1 - tmp.sum() / train_size
        train_errT[t] = err

        # test
        py = np.array(test_data["py"].tolist())
        py[py == "yes"] = 1
        py[py == "no"] = -1
        py = py.astype(int)
        test_py = test_py + py * alpha
        py = py.astype(str)
        py[test_py > 0] = "yes"
        py[test_py <= 0] = "no"
        test_data["py"] = pd.Series(py)
        tmp = test_data.apply(lambda row: 1 if row["y"] == row["py"] else 0, axis=1)
        err = 1 - tmp.sum() / test_size
        test_errT[t] = err
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(trainingError)
    ax1.plot(testingError)
    ax1.set_title("Each Tree Predictor")
    ax1.set_xlabel("Tree Number", fontsize=16)
    ax1.set_ylabel("Error Rate", fontsize=16)
    ax2.plot(train_errT)
    ax2.plot(test_errT)
    ax2.legend(["train", "test"])
    ax2.set_title("Combined Predictor")
    ax2.set_xlabel("Iteration", fontsize=16)
    f.savefig(f"ADA{T}_its.png")
