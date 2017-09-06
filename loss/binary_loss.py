import numpy as np
from loss import Loss


class BinaryLoss(Loss):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    @staticmethod
    def compute_loss_and_grad(Y, Y_hat):
        assert (np.all(Y.shape == Y_hat.shape))
        assert (Y.shape[0] == 1)
        m = Y.shape[1]
        logprobs = np.multiply(-np.log(Y_hat), Y) + np.multiply(-np.log(1-Y_hat), 1-Y)
        loss = 1.0 / m * np.nansum(logprobs)
        dY_hat = -np.divide(Y, Y_hat) + np.divide(1-Y, 1-Y_hat)
        return loss, dY_hat
