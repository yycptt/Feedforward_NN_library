import numpy as np
from activation import Activation


class LeakyReluActivation(Activation):
    def __init__(self, alpha):
        super(LeakyReluActivation, self).__init__()
        self.cache = None
        self.alpha = alpha

    def forward(self, z):
        self.cache = z
        a = np.array(z, copy=True)
        a[z <= 0] *= self.alpha
        return a

    def backward(self, da):
        dz = np.array(da, copy=True)
        dz[self.cache <= 0] *= self.alpha
        assert (dz.shape == self.cache.shape)
        return dz
