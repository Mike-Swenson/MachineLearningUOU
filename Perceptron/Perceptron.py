import pandas as pd
import numpy as np


class Perceptron:
    def __init__(self):
        self.learningRate = 0.1
        self.epochs = 10
        self.gamma = 0.1

    def standard(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        idx = np.arange(num_sample)
        for epochs in range(self.epochs):
            np.random.shuffle(idx)
            x = x[idx, :]
            y = y[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w = w + self.learningRate * y[i] * x[i]
        return w

    def voted(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        idx = np.arange(num_sample)
        c_list = np.array([])
        w_list = np.array([])
        c = 0
        for epochs in range(self.epochs):
            np.random.shuffle(idx)
            x = x[idx, :]
            y = y[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w_list = np.append(w_list, w)
                    c_list = np.append(c_list, c)
                    w = w + self.learningRate * y[i] * x[i]
                    c = 1
                else:
                    c = c + 1
        num = c_list.shape[0]
        w_list = np.reshape(w_list, (num, -1))
        return c_list, w_list

    def averageP(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        a = np.zeros(dim)
        idx = np.arange(num_sample)
        for epochs in range(self.epochs):
            np.random.shuffle(idx)
            x = x[idx, :]
            y = y[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w = w + self.learningRate * y[i] * x[i]
                a = a + w
        return a
