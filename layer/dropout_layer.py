import numpy as np
from layer import Layer


class DropoutLayer(Layer):
    def __init__(self):
        super(DropoutLayer, self).__init__()
        self.D = None
        self.keep_prob = 0.5

    def initialize(self):
        pass

    def forward(self, X, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.D = (np.random.rand(*X.shape) < self.keep_prob)
        return X * self.D / self.keep_prob

    def backward(self, dA):
        dZ = dA * self.D / self.keep_prob
        return dZ

    def getparams(self):
        return None

    def getgrads(self):
        return None

    def setparams(self, params):
        pass
