import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LMS:
    def __init__(self):
        self.method = 0
        self.lr = 0.25
        # Suggested by assignment
        self.threshold = 1e-5
        self.max_iter = 800

    def set_method(self, method):
        self.method = method

    def set_lr(self, lr):
        self.lr = lr

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    def optimize(self, x, y):
        if self.method == 0:
            return self.GD(x, y)
        elif self.method == 1:
            return self.stochastic(x, y)
        elif self.method == 2:
            return self.directEval(x, y)

    # x is augmented
    def GD(self, x, y):
        dim = x.shape[1]
        # update difference
        diff = 1
        # init w
        w = np.zeros([dim, 1])
        values = []
        it = 0
        while diff > self.threshold and it < self.max_iter:
            it = it + 1
            tmp = np.reshape(np.squeeze(np.matmul(x, w)) - y, (-1, 1))
            g = np.reshape(np.sum(np.transpose(tmp * x), axis=1), (-1, 1))
            delta = -self.lr * g
            w_new = w + delta
            diff = np.sqrt(np.sum(np.square(delta)))
            w = w_new
            tmp = np.reshape(np.squeeze(np.matmul(x, w)) - y, (-1, 1))
            functionVal = 0.5 * np.sum(np.square(tmp))
            values.append(functionVal)

        # save func_val iter plot
        fig = plt.figure()
        plt.xlabel("Step")
        plt.ylabel("Cost Function")
        plt.plot(values)
        plt.legend(["train"])
        fig.savefig(f"GradientDescent.png")
        return w

    def stochastic(self, x, y):
        dim = x.shape[1]
        n = x.shape[0]
        diff = 1
        w = np.zeros([dim, 1])
        functionVal = 1
        values = []
        it = 0
        while functionVal > self.threshold:
            it = it + 1
            idx = np.random.randint(n, size=1)
            x1 = x[idx]
            y1 = y[idx]
            g = np.sum(np.transpose((np.matmul(x1, w) - y1) * x1), axis=1)
            delta = -self.lr * np.reshape(g, (-1, 1))
            w_new = w + delta
            diff = np.sqrt(np.sum(np.square(delta)))
            w = w_new

            # Error
            # Remove axes of length one array
            tmp = np.reshape(np.squeeze(np.matmul(x, w)) - y, (-1, 1))

            # Code train video
            # Cost
            functionVal = 0.5 * np.sum(np.square(tmp))
            values.append(functionVal)

        # save func_val iter plot
        fig = plt.figure()
        plt.xlabel("Step")
        plt.ylabel("Cost Function")
        plt.plot(values)
        plt.legend(["train"])
        fig.savefig("SGD.png")

        return w

    def directEval(self, x, y):
        x_t = np.transpose(x)
        t1 = np.linalg.inv(np.matmul(x_t, x))
        t2 = np.matmul(x_t, y)
        return np.matmul(t1, t2)
