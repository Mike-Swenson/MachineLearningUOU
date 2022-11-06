import pandas as pd
import numpy as np
import Perceptron
import sys


trainInput = None
trainLabels = None
testInput = None
testLabels = None

precerptonObj = None


def processData():
    trainingData = pd.read_csv("./bank-note/train.csv", header=None)

    raw = trainingData.values
    columns = raw.shape[1]
    rows = raw.shape[0]

    trainInput = np.copy(raw)
    trainInput[:, columns - 1] = 1

    trainLabels = raw[:, columns - 1]
    trainLabels = 2 * trainLabels - 1

    test_data = pd.read_csv("./bank-note/test.csv", header=None)
    raw = test_data.values

    columns = raw.shape[1]
    rows = raw.shape[0]

    testInput = np.copy(raw)
    testInput[:, columns - 1] = 1

    testLabels = raw[:, columns - 1]

    # labels
    testLabels = 2 * testLabels - 1

    precerptonObj = Perceptron.Perceptron()
    return trainInput, trainLabels, testInput, testLabels, precerptonObj


def runPerceptron(
    perceptronSelect, trainInput, trainLabels, testInput, testLabels, precerptonObj
):
    if perceptronSelect == 0:
        weights = precerptonObj.standard(trainInput, trainLabels)
        weights = np.reshape(weights, (-1, 1))
        predictions = np.matmul(testInput, weights)
        predictions[predictions > 0] = 1
        predictions[predictions <= 0] = -1

        # Reshape (-1,1) basically lets numpy figure out the rows
        standardErr = (
            np.sum(np.abs(predictions - np.reshape(testLabels, (-1, 1))))
            / 2
            / testLabels.shape[0]
        )
        print(f"Standard Error: {standardErr}")
        print(f"Weighted Vector:\n {weights}")

    # voting
    elif perceptronSelect == 1:
        incorrectCounter, weightedVec = precerptonObj.voted(trainInput, trainLabels)
        incorrectCounter = np.reshape(incorrectCounter, (-1, 1))
        weightedVec = np.transpose(weightedVec)

        prod = np.matmul(testInput, weightedVec)
        prod[prod > 0] = 1
        prod[prod <= 0] = -1

        # Get weighted Results
        voted = np.matmul(prod, incorrectCounter)

        voted[voted > 0] = 1
        voted[voted <= 0] = -1
        err = (
            np.sum(np.abs(voted - np.reshape(testLabels, (-1, 1))))
            / 2
            / testLabels.shape[0]
        )

        print(f"Voted Error: {err}")
        print(f"Counters: {incorrectCounter}")
        print(f"Weighted Vector: {weightedVec}")

    elif perceptronSelect == 2:
        weights = precerptonObj.averageP(trainInput, trainLabels)
        weights = np.reshape(weights, (-1, 1))
        predictions = np.matmul(testInput, weights)
        predictions[predictions > 0] = 1
        predictions[predictions <= 0] = -1
        err = (
            np.sum(np.abs(predictions - np.reshape(testLabels, (-1, 1))))
            / 2
            / testLabels.shape[0]
        )
        print(f"Average Error:{err}")
        print(f"Weighted Vector: {weights}")


# 0: Standard
# 1: Voted
# 2: Average
if __name__ == "__main__":
    perceptronSelect = int(sys.argv[1])
    trainInput, trainLabels, testInput, testLabels, precerptonObj = processData()
    runPerceptron(
        perceptronSelect, trainInput, trainLabels, testInput, testLabels, precerptonObj
    )
