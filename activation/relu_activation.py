import numpy as np
from activation import Activation


class ReluActivation(Activation):
    def __init__(self):
        super(ReluActivation, self).__init__()
        self.cache = None

    def forward(self, z):
        self.cache = z
        return np.maximum(z, 0)

    def backward(self, da):
        dz = np.array(da, copy=True)
        dz[self.cache <= 0] = 0
        assert(dz.shape == self.cache.shape)
        return dz
