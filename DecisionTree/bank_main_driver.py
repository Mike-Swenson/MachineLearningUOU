# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:38:27 2022

@author: Michael Swenson
"""

import pandas as pd
import Tree_revised as idTree

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

numeric_features = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]
# this was super annoying
# Preprocess all data that is numeric

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
    # Preprocess all data that is numeric
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
    "duration": [0, 1],  # converted to binary
    "campaign": [0, 1],  # converted to binary
    "pdays": [0, 1],  # converted to binary
    "previous": [0, 1],  # converted to binary
    "poutcome": ["unknown", "other", "failure", "success"],
}

label = {"y": ["yes", "no"]}

# load and process Unknowns
bankTrainData = pd.read_csv("bank/train.csv", names=columns, dtype=types)


# convert numeric to binary
for c in numeric_features:
    median = bankTrainData[c].median()
    bankTrainData[c] = bankTrainData[c].apply(lambda x: 0 if x < median else 1)
# replace unknowns
keepUnknowns = False

unknown_features = ["job", "education", "contact", "poutcome"]
if keepUnknowns:
    for c in unknown_features:
        order = bankTrainData[c].value_counts().index.tolist()
        if order[0] != "unknown":
            replace = order[0]
        else:
            replace = order[1]
        bankTrainData[c] = bankTrainData[c].apply(
            lambda x: replace if x == "unknown" else x
        )
else:
    bankTrainData = bankTrainData[bankTrainData.poutcome != "unknown"]
train_size = len(bankTrainData.index)

# load and process Unknowns
test_data = pd.read_csv("bank/test.csv", names=columns, dtype=types)
for c in numeric_features:
    median = test_data[c].median()
    test_data[c] = test_data[c].apply(lambda x: 0 if x < median else 1)
if keepUnknowns:
    for c in unknown_features:
        order = test_data[c].value_counts().index.tolist()
        if order[0] != "unknown":
            replace = order[0]
        else:
            replace = order[1]
        test_data[c] = test_data[c].apply(lambda x: replace if x == "unknown" else x)
else:
    test_data = test_data[test_data.poutcome != "unknown"]
test_size = len(test_data.index)

train_acc = [[0 for x in range(16)] for y in range(3)]
test_acc = [[0 for x in range(16)] for y in range(3)]

for feature_selection in range(3):
    for max_depth in range(16):
        # Nice to see if we're stuck or just working
        print(f"Processing Depth: {max_depth}")

        # same pattern as car stuff
        idTree = idTree.ID3(
            feature_selection=feature_selection, max_depth=max_depth + 1
        )
        tree = idTree.makeTree(bankTrainData, features, label)
        bankTrainData["leafLabel"] = idTree.predict(tree, bankTrainData)
        train_acc[feature_selection][max_depth] = (
            bankTrainData.apply(
                lambda row: 1 if row["label"] == row["leafLabel"] else 0, axis=1
            ).sum()
            / train_size
        )

        test_data["leafLabel"] = idTree.predict(tree, test_data)
        test_acc[feature_selection][max_depth] = (
            test_data.apply(
                lambda row: 1 if row["label"] == row["leafLabel"] else 0, axis=1
            ).sum()
            / test_size
        )
print("train_acc:", train_acc)
print("test_acc:", test_acc)
