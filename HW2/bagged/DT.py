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

    def set_feature(self, feature):
        self.feature = feature

    def set_children(self, children):
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

    # feature_selection: IG ME GINI 0 1 2
    def __init__(self, feature_selection=0, max_depth=10):
        self.feature_selection = feature_selection
        self.max_depth = max_depth

    def set_feature_selection(self, feature_selection):
        self.feature_selection = feature_selection

    def set_max_depth(self, max_depth):
        self.max_depth = max_depth

    def IG(self, dataset, label):
        labelName, labelValue = list(label.items())[0]
        total = len(dataset.index)
        if total == 0:
            return 0
        entropy = 0
        for v in labelValue:
            p = len(dataset[dataset[labelName] == v]) / total
            if p != 0:
                entropy += -p * math.log2(p)
        return entropy

    def ME(self, dataset, label):
        labelName, labelValue = list(label.items())[0]
        total = len(dataset.index)
        if total == 0:
            return 0
        max_p = 0
        for v in labelValue:
            p = len(dataset[dataset[labelName] == v]) / total
            max_p = max(max_p, p)
        return 1 - max_p

    def gini(self, dataset, label):
        labelName, labelValue = list(label.items())[0]
        total = len(dataset.index)
        if total == 0:
            return 0
        sqrsum = 0
        for v in labelValue:
            p = len(dataset[dataset[labelName] == v]) / total
            sqrsum += p ** 2
        return 1 - sqrsum

    def split(self, cur):
        nodeList = []
        features = cur["features"]
        label = cur["label"]
        dtNode = cur["dtNode"]
        dataset = cur["dataset"]

        measure = None
        if self.feature_selection == 0:
            measure = self.IG
        elif self.feature_selection == 1:
            measure = self.ME
        elif self.feature_selection == 2:
            measure = self.gini

        total = len(dataset.index)

        labelName, labelValue = list(label.items())[0]
        if total > 0:
            majority_label = dataset[labelName].value_counts().idxmax()

        stat = measure(dataset, label)
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
            gain = 0
            for v in fv:
                subset = dataset[dataset[fn] == v]
                p = len(subset.index) / total
                gain += p * measure(subset, label)
            gain = stat - gain
            if gain > max_gain:
                max_gain = gain
                max_fn = fn

        children = {}
        dtNode.set_feature(max_fn)
        # remaining features
        rf = copy.deepcopy(features)
        rf.pop(max_fn, None)
        # split dataset
        for v in features[max_fn]:
            childNode = TreeNode()
            childNode.set_depth(dtNode.get_depth() + 1)
            childNode.set_label(majority_label)
            children[v] = childNode
            pNode = {
                "dataset": dataset[dataset[max_fn] == v],
                "features": copy.deepcopy(rf),
                "label": label,
                "dtNode": childNode,
            }
            nodeList.append(pNode)
        dtNode.set_children(children)
        # return processing nodes
        return nodeList

    ## generate decision tree
    def makeTree(self, dataset, features, label):
        Q = []
        dtRoot = TreeNode()
        dtRoot.set_depth(0)
        # processing node root
        root = {
            "dataset": dataset,
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
