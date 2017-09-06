import numpy as np
from activation import Activation


class TanhActivation(Activation):
    def __init__(self):
        super(TanhActivation, self).__init__()
        self.cache = None

    def forward(self, z):
        self.cache = z
        return np.tanh(z)

    def backward(self, da):
        dz = da * (1.0 / np.square(np.cosh(self.cache)))
        assert (dz.shape == self.cache.shape)
        return dz
