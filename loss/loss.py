import abc


class Loss(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def compute_loss_and_grad(Y, Y_hat):
        pass
