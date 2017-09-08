import numpy as np
from layer import Layer


class DropoutLayer(Layer):
    def __init__(self, keep_prob):
        super(DropoutLayer, self).__init__()
        self.D = None
        self.keep_prob = keep_prob

    def initialize(self):
        pass

    def forward(self, X, train=True):
        if train:
            self.D = (np.random.rand(*X.shape) < self.keep_prob)
            return X * self.D / self.keep_prob
        else:
            return X

    def backward(self, dA):
        dZ = dA * self.D / self.keep_prob
        return dZ

    def getparams(self):
        return None

    def getgrads(self):
        return None

    def updateparams(self, params):
        pass

    def get_l2_loss(self):
        return 0
