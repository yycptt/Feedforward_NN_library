class Model(object):
    def __init__(self, layers, optimizer, loss):
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss

    def initialize(self):
        for layer in self.layers:
            layer.initialize()
        self.optimizer.initialize()

    def predict(self, A):
        for layer in self.layers:
            A = layer.forward(A, train=False)
        return A

    def train(self, A, Y):
        loss = 0
        for layer in self.layers:
            A = layer.forward(A, train=True)
            loss += layer.get_l2_loss()
        Y_hat = A
        cost, dA = self.loss.compute_loss_and_grad(Y, Y_hat)
        loss += cost
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        self.optimizer.update()
        return loss
