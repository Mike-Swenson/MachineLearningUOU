# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 00:58:06 2022

@author: Felix
"""

import pandas as pd
import math
import copy

# Followed Example from slides in class
class TreeNode:
    def __init__(self):
        self.feature = None
        self.children = None
        self.depth = -1
        self.isLeaf = False
        self.label = None

    def setFeature(self, feature):
        self.feature = feature

    def setChildren(self, children):
        self.children = children

    def get_depth(self):
        return self.depth

    def set_depth(self, depth):
        self.depth = depth

    def set_leaf(self):
        self.isLeaf = True

    def is_leaf(self):
        return self.isLeaf

    def set_label(self, label):
        self.label = label


class ID3:
    # Again from class slides
    def __init__(self, feature_selection=0, max_depth=10):
        self.feature_selection = feature_selection
        self.max_depth = max_depth

    @property
    def setFeature_selection(self, feature_selection):
        self.feature_selection = feature_selection

    @property
    def depth(self, max_depth):
        self.max_depth = max_depth

    def IG(self, dataset, label):
        total = len(dataset.index)
        if total == 0:
            return 0
        # featuresList List of Labels
        # label Column label
        columnLabel, featuresList = list(label.items())[0]
        entropy = 0
        # for each label in All labels
        for value in featuresList:
            p = len(dataset[dataset[columnLabel] == value]) / total
            if p != 0:
                entropy += -p * math.log2(p)
        return entropy

    def ME(self, dataset, label):
        columnLabel, featuresList = list(label.items())[0]
        total = len(dataset.index)
        if total == 0:
            return 0
        max_p = 0
        for value in featuresList:
            p = len(dataset[dataset[columnLabel] == value]) / total
            max_p = max(max_p, p)
        return 1 - max_p

    def gini(self, dataset, label):
        columnLabel, featuresList = list(label.items())[0]
        total = len(dataset.index)
        if total == 0:
            return 0
        runningSquare = 0
        for value in featuresList:
            p = len(dataset[dataset[columnLabel] == value]) / total
            runningSquare += p ** 2
        return 1 - runningSquare

    ## generate decision tree
    def makeTree(self, dataset, features, label):
        dfs = []
        treeRoot = TreeNode()
        treeRoot.set_depth(0)
        # processing node root
        root = {
            "dataset": dataset,
            "features": features,
            "label": label,
            "dtNode": treeRoot,
        }
        dfs.append(root)
        while len(dfs) > 0:
            cur = dfs.pop(0)
            nodes = self.split(cur)
            for node in nodes:
                dfs.append(node)
        return treeRoot

    def split(self, currentNode):
        nodeList = []
        dataset = currentNode["dataset"]
        label = currentNode["label"]
        node = currentNode["dtNode"]
        features = currentNode["features"]

        # python doesn't have a switch
        metric = None
        if self.feature_selection == 0:
            metric = self.IG
        elif self.feature_selection == 1:
            metric = self.ME
        elif self.feature_selection == 2:
            metric = self.gini
        total = len(dataset.index)

        # featuresList List of Labels
        # columnLabel Column label
        columnLabel, featuresList = list(label.items())[0]
        if total > 0:
            # Could maybe replace with collections Counter most_common
            majority_label = dataset[columnLabel].value_counts().idxmax()
        setEnt = metric(dataset, label)

        # Are we at a leaf node?
        if (
            setEnt == 0
            or node.get_depth() == self.max_depth
            or len(features.items()) == 0
        ):
            node.set_leaf()
            if total > 0:
                node.set_label(majority_label)
            return nodeList
        maxFeatureEnt = -1
        maxFeatureName = None
        # select feature which results in maximum gain
        # Loop through available label features to choose the maximum gain or minimum entropy
        for featureName, featureVal in features.items():
            entropy = 0
            # Calculate and find the max, could shorten with a PQ
            for value in featureVal:
                subset = dataset[dataset[featureName] == value]
                p = len(subset.index) / total
                entropy += p * metric(subset, label)
            entropy = setEnt - entropy
            if entropy > maxFeatureEnt:
                maxFeatureEnt = entropy
                maxFeatureName = featureName
        children = {}
        node.setFeature(maxFeatureName)
        # Required to actually get the list
        rf = copy.deepcopy(features)
        rf.pop(maxFeatureName, None)
        # We've found the best feature now create children based on the different options
        for value in features[maxFeatureName]:
            childNode = TreeNode()
            childNode.set_depth(node.get_depth() + 1)
            childNode.set_label(majority_label)
            children[value] = childNode
            pNode = {
                "dataset": dataset[dataset[maxFeatureName] == value],
                "features": copy.deepcopy(rf),
                "label": label,
                "dtNode": childNode,
            }
            nodeList.append(pNode)
        node.setChildren(children)
        # Return all children and recurse in the tree
        return nodeList

    ########################################################################
    # Predict

    # Basically just finding the leaf node with the same feature
    def predict(self, dt, test_data):
        return test_data.apply(lambda row: self.predictRecurse(dt, row), axis=1)

    def predictRecurse(self, dt, test_data):
        p = dt
        while not p.is_leaf():
            p = p.children[test_data[p.feature]]
        return p.label
