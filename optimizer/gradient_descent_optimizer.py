from optimizer import Optimizer


class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate, layers):
        super(GradientDescentOptimizer, self).__init__(learning_rate, layers)

    def initialize(self):
        pass

    def update(self):
        for layer in self.layers:
            grads = layer.getgrads()
            if grads is None:
                continue
            grads = list(grads)
            for j, grad in enumerate(grads):
                grads[j] = -self.learning_rate*grads[j]
            layer.updateparams(grads)
