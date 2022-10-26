import pandas as pd
import math
import copy
import numpy as np
import warnings

warnings.simplefilter("ignore")

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

    def set_label(self, label):
        self.label = label

    def is_leaf(self):
        return self.isLeaf


class weightedID3:
    # Again from class slides
    def __init__(self, feature_selection=0, max_depth=10):
        self.feature_selection = feature_selection
        self.max_depth = max_depth

    @property
    def setFeature_selection(self, feature_selection):
        self.feature_selection = feature_selection

    def set_max_depth(self, max_depth):
        self.max_depth = max_depth

    # Started using numpy for ease of use, speed, and LA tools
    # TODO
    def IG(self, dataset, label, weights):
        labelName, labelValue = list(label.items())[0]
        total = np.sum(weights)
        column = np.array(dataset[labelName].tolist())
        if total == 0:
            return 0
        entropy = 0

        # NP integration for speed and LA
        for v in labelValue:
            # myRows = df[df['Unnamed: 5'].map( lambda x: x == 'Peter' )].index.tolist() numpy/pandas/python fighting
            w = weights[column == v]
            p = np.sum(w) / total
            if p != 0:
                entropy += -p * math.log2(p)
        return entropy

    def ME(self, dataset, label, weights):
        labelName, labelValue = list(label.items())[0]
        total = np.sum(weights)
        column = np.array(dataset[labelName].tolist())
        if total == 0:
            return 0
        max_p = 0
        for v in labelValue:
            w = weights[column == v]
            # Convert to NP where usable
            p = np.sum(w) / total
            max_p = max(max_p, p)
        return 1 - max_p

    def gini(self, dataset, label, weights):
        labelName, labelValue = list(label.items())[0]
        total = np.sum(weights)
        column = np.array(dataset[labelName].tolist())
        if total == 0:
            return 0
        sqrsum = 0
        for v in labelValue:
            w = weights[column == v]
            p = np.sum(w) / total
            sqrsum += p ** 2
        return 1 - sqrsum

    # moved in here
    def get_majority(self, dataset, label, weights):
        labelName, labelValue = list(label.items())[0]
        majority_label = None
        max_sum = -1
        column = np.array(dataset[labelName].tolist())
        for v in labelValue:
            w = weights[column == v]
            weight_sum = np.sum(w)
            if weight_sum > max_sum:
                majority_label = v
                max_sum = weight_sum

        return majority_label

    def split(self, currentNode):
        nodeList = []

        features = currentNode["features"]
        label = currentNode["label"]
        dtNode = currentNode["dtNode"]
        dataset = currentNode["dataset"]
        weights = currentNode["weights"]

        measure = None
        if self.feature_selection == 0:
            measure = self.IG
        elif self.feature_selection == 1:
            measure = self.ME
        elif self.feature_selection == 2:
            measure = self.gini

        total = sum(weights)
        majority_label = self.get_majority(dataset, label, weights)

        stat = measure(dataset, label, weights)
        # pure or achieve max depth or no remaining features
        if (
            stat == 0
            or dtNode.get_depth() == self.max_depth
            or len(features.items()) == 0
        ):
            dtNode.set_leaf()
            if total > 0:
                dtNode.set_label(majority_label)
            return nodeList

        max_gain = -1
        max_fn = None
        # select feature which results in maximum gain
        for fn, fv in features.items():
            column = np.array(dataset[fn].tolist())
            gain = 0
            for v in fv:
                w = weights[column == v]
                sub_weights = w
                p = np.sum(sub_weights) / total
                subset = dataset[dataset[fn] == v]
                gain += p * measure(subset, label, sub_weights)
            gain = stat - gain
            if gain > max_gain:
                max_gain = gain
                max_fn = fn

        children = {}
        dtNode.setFeature(max_fn)
        # Required to actually get the list
        rf = copy.deepcopy(features)
        rf.pop(max_fn, None)
        # We've found the best feature now create children based on the different options
        column = np.array(dataset[max_fn].tolist())
        for v in features[max_fn]:
            childNode = TreeNode()
            childNode.set_depth(dtNode.get_depth() + 1)
            childNode.set_label(majority_label)
            children[v] = childNode
            w = weights[column == v]
            pNode = {
                "dataset": dataset[dataset[max_fn] == v],
                "weights": w,
                "features": copy.deepcopy(rf),
                "label": label,
                "dtNode": childNode,
            }
            nodeList.append(pNode)
        dtNode.setChildren(children)
        return nodeList

    def makeTree(self, dataset, features, label, weights):
        Q = []
        dtRoot = TreeNode()
        dtRoot.set_depth(0)
        # processing node root
        root = {
            "dataset": dataset,
            "weights": weights,
            "features": features,
            "label": label,
            "dtNode": dtRoot,
        }
        Q.append(root)
        while len(Q) > 0:
            cur = Q.pop(0)
            nodes = self.split(cur)
            for node in nodes:
                Q.append(node)
        return dtRoot

    def predictRecurse(self, dt, test_data):
        p = dt
        while not p.is_leaf():
            p = p.children[test_data[p.feature]]
        return p.label

    def predict(self, dt, test_data):
        return test_data.apply(lambda row: self.predictRecurse(dt, row), axis=1)
