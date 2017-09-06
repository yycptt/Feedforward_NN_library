from activation import Activation


class LinearActivation(Activation):
    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, z):
        return z

    def backward(self, da):
        return da
