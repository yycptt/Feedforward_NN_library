import numpy as np
from layer import Layer


class FullyConnectedLayer(Layer):
    def __init__(self, num_in, num_out, act, l2_lambda):
        super(FullyConnectedLayer, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.activation = act
        self.Wgrad = None
        self.bgrad = None
        self.cache = None
        self.W = None
        self.b = None
        self.l2_lambda = l2_lambda
        self.m = None

    def initialize(self):
        self.W = np.random.randn(self.num_out, self.num_in) * 0.01
        self.b = np.zeros((self.num_out, 1))

    def forward(self, X, train=True):
        self.m = X.shape[1]
        Z = np.dot(self.W, X) + self.b
        A = self.activation.forward(Z)
        self.cache = X
        return A

    def backward(self, dA):
        dZ = self.activation.backward(dA)
        dX = np.dot(self.W.T, dZ)
        self.Wgrad = 1.0/self.m * np.dot(dZ, self.cache.T)
        self.bgrad = 1.0/self.m * np.sum(dZ, axis=1, keepdims=True)
        self.Wgrad += 1.0 * self.l2_lambda / self.m * self.W
        return dX

    def getparams(self):
        return self.W, self.b

    def getgrads(self):
        return self.Wgrad, self.bgrad

    def updateparams(self, params):
        self.W = self.W + params[0]
        self.b = self.b + params[1]

    def get_l2_loss(self):
        return self.l2_lambda / 2.0 / self.m * np.sum(np.square(self.W))
