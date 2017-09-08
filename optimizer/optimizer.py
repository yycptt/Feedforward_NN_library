import abc


class Optimizer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, learning_rate, layers):
        self.learning_rate = learning_rate
        self.layers = layers

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass
