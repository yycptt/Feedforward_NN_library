import numpy as np
from loss import Loss


class CrossEntropyLoss(Loss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    @staticmethod
    def compute_loss_and_grad(Y, Y_hat):
        assert (np.all(Y.shape == Y_hat.shape))
        m = Y.shape[1]
        loss = 1.0 / m * np.nansum(-np.multiply(Y, np.log(Y_hat)))
        dY_hat = -np.divide(Y, Y_hat)
        return loss, dY_hat
