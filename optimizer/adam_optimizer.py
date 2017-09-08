import numpy as np
from optimizer import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate, layers, beta1, beta2):
        super(AdamOptimizer, self).__init__(learning_rate, layers)
        self.beta1 = beta1
        self.beta2 = beta2
        self.v = []
        self.s = []
        self.t = 0

    def initialize(self):
        for i, layer in enumerate(self.layers):
            self.v.append([])
            self.s.append([])
            params = layer.getparams()
            if params is None:
                continue
            for param in params:
                self.v[i].append(np.zeros(param.shape))
                self.s[i].append(np.zeros(param.shape))

    def update(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            grads = layer.getgrads()
            if grads is None:
                continue
            grads = list(grads)
            for j, grad in enumerate(grads):
                self.v[i][j] = self.v[i][j]*self.beta1 + (1-self.beta1) * grad
                v_corrected = self.v[i][j] / (1.0 - self.beta1**self.t)
                self.s[i][j] = self.s[i][j]*self.beta2 + (1-self.beta2) * np.square(grad)
                s_corrected = self.s[i][j] / (1.0 - self.beta2**self.t)
                grads[j] = - self.learning_rate * v_corrected / (np.sqrt(s_corrected) + 1e-7)
            layer.updateparams(grads)
