import numpy as np
from activation import Activation


class SigmoidActivation(Activation):
    def __init__(self):
        super(SigmoidActivation, self).__init__()
        self.cache = None

    def forward(self, z):
        a = 1.0 / (1 + np.exp(-z))
        self.cache = a
        return a

    def backward(self, da):
        dz = da * self.cache * (1 - self.cache)
        assert(dz.shape == self.cache.shape)
        return dz
