import abc


class Activation(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def forward(self, z):
        pass

    @abc.abstractmethod
    def backward(self, da):
        pass
