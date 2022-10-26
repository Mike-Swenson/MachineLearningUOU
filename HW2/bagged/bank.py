#%%
import pandas as pd
import DT as dt
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
    # load train data
    train_data = pd.read_csv("./bank/train.csv", names=columns, dtype=types)
    train_size = len(train_data.index)
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
        "balance": [0, 1],  # converted to binary
        "housing": ["yes", "no"],
        "loan": ["yes", "no"],
        "contact": ["unknown", "telephone", "cellular"],
        "day": [0, 1],  # converted to binary,
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

    train_err = [0 for x in range(T)]
    test_err = [0 for x in range(T)]
    train_py = np.array([0 for x in range(train_size)])
    test_py = np.array([0 for x in range(test_size)])

    for t in range(T):

        bootstrapped = train_data.sample(frac=0.5, replace=True, random_state=t)
        # ID3
        dt_generator = dt.weightedID3(feature_selection=0, max_depth=17)
        decision_tree = dt_generator.makeTree(bootstrapped, features, label)

        # train
        py = dt_generator.predict(decision_tree, train_data)
        py = np.array(py.tolist())
        py[py == "yes"] = 1
        py[py == "no"] = -1
        py = py.astype(int)
        train_py = train_py + py
        py = py.astype(str)
        py[train_py > 0] = "yes"
        py[train_py <= 0] = "no"
        train_data["py"] = pd.Series(py)

        acc = (
            train_data.apply(
                lambda row: 1 if row["y"] == row["py"] else 0, axis=1
            ).sum()
            / train_size
        )
        err = 1 - acc
        train_err[t] = err
        # test
        py = dt_generator.predict(decision_tree, test_data)
        py = np.array(py.tolist())
        py[py == "yes"] = 1
        py[py == "no"] = -1
        py = py.astype(int)
        test_py = test_py + py
        py = py.astype(str)
        py[test_py > 0] = "yes"
        py[test_py <= 0] = "no"
        test_data["py"] = pd.Series(py)
        acc = (
            test_data.apply(lambda row: 1 if row["y"] == row["py"] else 0, axis=1).sum()
            / test_size
        )
        err = 1 - acc
        test_err[t] = err
        print("t: ", t, "train_err: ", train_err[t], "test_err: ", test_err[t])

    fig = plt.figure()
    fig.suptitle("Bagged Decision Tree")
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Error Rate", fontsize=16)
    plt.plot(train_err, "b")
    plt.plot(test_err, "r")
    plt.legend(["train", "test"])
    fig.savefig("bdt.png")
    train_py = np.array([0 for x in range(train_size)])
    test_py = np.array([0 for x in range(test_size)])

    print("Print total iterations: ", T)

    for t in range(T):
        print("iteration: ", t)
        # sample with replace
        bootstrapped = train_data.sample(frac=0.5, replace=True, random_state=t)
        # ID3
        dt_generator = dt.weightedID3(feature_selection=0, max_depth=17)
        # get decision tree
        decision_tree = dt_generator.makeTree(bootstrapped, features, label)

        ## predict
        # train
        py = dt_generator.predict(decision_tree, train_data)
        py = np.array(py.tolist())
        py[py == "yes"] = 1
        py[py == "no"] = -1
        py = py.astype(int)
        train_py = train_py + py
        py = py.astype(str)
        py[train_py > 0] = "yes"
        py[train_py <= 0] = "no"
        train_data["py"] = pd.Series(py)

        acc = (
            train_data.apply(
                lambda row: 1 if row["y"] == row["py"] else 0, axis=1
            ).sum()
            / train_size
        )
        err = 1 - acc
        train_err[t] = err
        # test
        py = dt_generator.predict(decision_tree, test_data)
        py = np.array(py.tolist())
        py[py == "yes"] = 1
        py[py == "no"] = -1
        py = py.astype(int)
        test_py = test_py + py
        py = py.astype(str)
        py[test_py > 0] = "yes"
        py[test_py <= 0] = "no"
        test_data["py"] = pd.Series(py)
        acc = (
            test_data.apply(lambda row: 1 if row["y"] == row["py"] else 0, axis=1).sum()
            / test_size
        )
        err = 1 - acc
        test_err[t] = err

    fig = plt.figure()
    fig.suptitle("Bagged Decision Tree")
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Error Rate", fontsize=16)
    plt.plot(train_err, "b")
    plt.plot(test_err, "r")
    plt.legend(["train", "test"])
    fig.savefig(f"bagged{T}.png")
