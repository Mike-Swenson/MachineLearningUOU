import numpy as np


class NN:
    def __init__(self, width, select=0):
        # input
        self.input_dim = width[0]
        # hidden layers inbetween
        # output
        self.output_dim = width[-1]
        self.learningRate = 0.1
        self.d = 0.1
        self.epoch = 100
        self.gamma = 0.1

        # width including input and output
        self.width = width
        self.layers = len(width)
        # weights
        self.w = [None for _ in range(self.layers)]
        # derivative weights
        # storing for "cache speed up"
        self.partialW = [None for _ in range(self.layers)]

        # Could use list comprehension here for speed up
        # Initialization
        for i in range(1, self.layers - 1):
            # Initialize the edge weights with random numbers generated from the std gaussian distro
            if select == 0:
                wi = np.random.normal(0, 1, (self.width[i] - 1, self.width[i - 1]))
                self.w[i] = wi
                self.partialW[i] = np.zeros([self.width[i] - 1, self.width[i - 1]])
            else:
                wi = np.zeros((self.width[i] - 1, self.width[i - 1]))
                self.w[i] = wi
                self.partialW[i] = np.zeros([self.width[i] - 1, self.width[i - 1]])

        #  redundandant, could probably add this to above also backwards
        i = self.layers - 1
        if select == 1:
            wi = np.zeros((self.width[i] - 1, self.width[i - 1]))
            self.w[i] = wi
            self.partialW[i] = np.zeros([self.width[i], self.width[i - 1]])
            self.nodes = [np.ones([self.width[i], 1]) for i in range(self.layers)]
        else:
            wi = np.random.normal(0, 1, (self.width[i], self.width[i - 1]))
            self.w[i] = wi
            self.partialW[i] = np.zeros([self.width[i], self.width[i - 1]])
            self.nodes = [np.ones([self.width[i], 1]) for i in range(self.layers)]

    def train(self, x, y):
        num_sample = x.shape[0]
        idx = np.arange(num_sample)
        for t in range(self.epoch):
            # Need to shuffle before every epoch
            np.random.shuffle(idx)
            x = x[idx, :]
            y = y[idx]
            for i in range(num_sample):
                self.predict_propagate(
                    x[i, :].reshape([self.input_dim, 1]),
                    y[i, :].reshape([self.output_dim, 1]),
                )

                # Schedule of learning rate
                learningRate = self.gamma / (1 + self.gamma / self.d * t)
                self.update_w(learningRate)

    def update_w(self, learningRate):
        for i in range(1, self.layers):
            self.w[i] = self.w[i] - self.learningRate * self.partialW[i]

    # predict calculates the Activation output and assigns
    def predict(self, x):
        # input
        self.nodes[0] = x
        for i in range(1, self.layers - 1):
            self.nodes[i][:-1, :] = self.sigmoid(
                np.matmul(self.w[i], self.nodes[i - 1]).reshape([-1, 1])
            )
        # output
        i = self.layers - 1
        self.nodes[i] = np.matmul(self.w[i], self.nodes[i - 1])

    # https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    # Was having overflow issues with the naive implementation
    def sigmoid(self, x):
        return np.exp(-np.logaddexp(0, -x))

    # Backwards calculates the partial weights
    # This is the back propagation, should go back and use this to answer #2 written
    # have to consider the constant nodes that are just bias
    def adjust(self, y):
        # output
        # Basically just cancel out: dz/dx = dz/dy1 * dy1/dx + (...)
        # dLdy
        dLdz = self.nodes[-1] - y
        nk = self.width[-1]
        dzdw = np.transpose(np.tile(self.nodes[-2], [1, nk]))
        self.partialW[-1] = dLdz * dzdw
        # dydz
        dzdz = self.w[-1][:, :-1]

        # Calculate backwards derivatives
        for i in reversed(range(1, self.layers - 1)):
            nk = self.width[i] - 1
            # ignore bias term in k
            # pointer, pointee, whose weight affects whom
            z_in = self.nodes[i - 1]
            z_out = self.nodes[i][:-1]

            dadw = np.transpose(np.tile(z_in, [1, nk]))
            # Advantage of sigmoid
            dzdw = z_out * (1 - z_out) * dadw

            # Derivative of Loss func (Y/L)
            dLdz = np.matmul(np.transpose(dzdz), dLdz)
            dLdw = dLdz * dzdw
            self.partialW[i] = dLdw

            # update
            dzdz = z_out * (1 - z_out) * self.w[i]
            # arrange
            dzdz = dzdz[:, :-1]

    def predict_propagate(self, x, y):
        self.predict(x)
        self.adjust(y)

    def fit(self, x):
        num_sample = x.shape[0]
        l = []
        for i in range(num_sample):
            self.predict(x[i, :].reshape(self.input_dim))
            y = self.nodes[-1]
            l.append(np.transpose(y))
        y_pred = np.concatenate(l, axis=0)
        return y_pred
