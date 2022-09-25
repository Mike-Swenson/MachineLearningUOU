# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 01:03:28 2022

@author: Michael Swenson
"""

import Tree_revised as idTree
import pandas as pd


# Have to specify the types because the bank data is not strings and is split on median
types = {
    "buying": str,
    "maint": str,
    "doors": str,
    "persons": str,
    "lug_boot": str,
    "safety": str,
    "label": str,
}

features = {
    "buying": ["vhigh", "high", "med", "low"],
    "maint": ["vhigh", "high", "med", "low"],
    "doors": ["2", "3", "4", "5more"],
    "persons": ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety": ["low", "med", "high"],
}

headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
label = {"label": ["unacc", "acc", "good", "vgood"]}


# Switched to Pandas because it was too hard to track / split data with labels/tuples etc
carTrainData = pd.read_csv("car/train.csv", names=headers, dtype=types)
trainingSamples = len(carTrainData.index)


carTestData = pd.read_csv("car/test.csv", names=headers, dtype=types)
testSamples = len(carTestData.index)


# Lambdas for driving the iterative 1-6 depth of the tree
# And we can vary the metric is one go
# Two of them for ease of reporting
trainingErr = [[0 for x in range(6)] for y in range(3)]
testErr = [[0 for x in range(6)] for y in range(3)]


# Create in Order IG, ME, GINI trainers
for feature_selection in range(3):
    # Iterate from 0 - 5 / 1-6 depths
    for max_depth in range(6):
        tree = idTree.ID3(feature_selection=feature_selection, max_depth=max_depth + 1)
        dTree = tree.makeTree(carTrainData, features, label)

        # Check how many labels are equal on the leaf nodes then sum and normalize
        carTrainData["leafLabel"] = tree.predict(dTree, carTrainData)
        trainingErr[feature_selection][max_depth] = (
            carTrainData.apply(
                lambda row: 1 if row["label"] == row["leafLabel"] else 0, axis=1
            ).sum()
            / trainingSamples
        )
        # Do the same thing for the test data
        carTestData["leafLabel"] = tree.predict(dTree, carTestData)
        testErr[feature_selection][max_depth] = (
            carTestData.apply(
                lambda row: 1 if row["label"] == row["leafLabel"] else 0, axis=1
            ).sum()
            / testSamples
        )
print("Training Report:\n", trainingErr)
print()
print("Test Data Report:", testErr)
