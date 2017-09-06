import numpy as np
from loss import Loss


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    @staticmethod
    def compute_loss_and_grad(Y, X):
        assert (np.all(Y.shape == X.shape))
        m = Y.shape[1]
        temp = np.exp(X)
        Y_hat = np.divide(temp, np.sum(temp, axis=0))
        loss = 1.0 / m * np.nansum(-np.multiply(Y, np.log(Y_hat)))
        dX = Y_hat - Y
        return loss, dX
