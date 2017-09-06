import numpy as np
from loss import Loss


class MSELoss(Loss):
    def __init__(self):
        super(MSELoss, self).__init__()

    @staticmethod
    def compute_loss_and_grad(Y, Y_hat):
        assert (np.all(Y.shape == Y_hat.shape))
        m = Y.shape[1]

        loss = 1.0 / m * np.sum(np.square(Y_hat - Y))
        dY_hat = 0.5 * (Y_hat - Y)
        return loss, dY_hat
