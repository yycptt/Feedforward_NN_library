import numpy as np
from layer import Layer


class FullyConnectedLayer(Layer):
    def __init__(self, num_in, num_out, act):
        super(FullyConnectedLayer, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.activation = act
        self.Wgrad = None
        self.bgrad = None
        self.cache = None
        self.W = None
        self.b = None

    def initialize(self):
        self.W = np.random.randn(self.num_out, self.num_in) * 0.01
        self.b = np.zeros((self.num_out, 1))

    def forward(self, X):
        Z = np.dot(self.W, X) + self.b
        A = self.activation.forward(Z)
        self.cache = X
        return A

    def backward(self, dA):
        dZ = self.activation.backward(dA)
        m = dZ.shape[1]
        dX = np.dot(self.W.T, dZ)
        self.Wgrad = 1.0/m * np.dot(dZ, self.cache.T)
        self.bgrad = 1.0/m * np.sum(dZ, axis=1, keepdims=True)
        return dX

    def getparams(self):
        return self.W, self.b

    def getgrads(self):
        return self.Wgrad, self.bgrad

    def setparams(self, params):
        self.W = params[0]
        self.b = params[1]
