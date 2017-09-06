import abc


class Layer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def forward(self, X):
        pass

    @abc.abstractmethod
    def backward(self, dA):
        pass

    @abc.abstractmethod
    def getparams(self):
        pass

    @abc.abstractmethod
    def getgrads(self):
        pass

    @abc.abstractmethod
    def setparams(self, params):
        pass
