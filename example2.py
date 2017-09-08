from activation.relu_activation import ReluActivation
from activation.linear_activation import LinearActivation
from layer.fully_connected_layer import FullyConnectedLayer
from layer.dropout_layer import DropoutLayer
from loss.softmax_cross_entropy_loss import SoftmaxCrossEntropyLoss
from optimizer.adam_optimizer import AdamOptimizer
from model import Model

import numpy as np
from sklearn.datasets import load_digits
import time


def random_mini_batches(X, Y, mini_batch_size=64):
    np.random.seed(int(time.time()))
    m = X.shape[1]
    mini_batches = []
    out_dim = Y.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((out_dim, m))

    num_complete_minibatches = int(np.floor(m / mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


if __name__ == '__main__':
    data = load_digits()
    m = data.data.shape[0]
    X = data.data.T
    Y = np.reshape(data.target, (1, m))
    out_dim = Y.shape[0]
    permutation = list(np.random.permutation(m))
    X = X[:, permutation]
    Y = Y[:, permutation].reshape((out_dim, m))

    X = X / 16.0
    Y_onehot = np.zeros((10, m))
    Y_onehot[Y[0, :], range(m)] = 1

    m_train = int(m * 0.8)
    X_train = X[:, :m_train]
    Y_train = Y_onehot[:, :m_train]
    X_test = X[:, m_train:]
    Y_test = Y_onehot[:, m_train:]
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    layer1 = FullyConnectedLayer(64, 30, ReluActivation(), 0.01)
    dropout = DropoutLayer(0.9)
    layer2 = FullyConnectedLayer(30, 10, LinearActivation(), 0.01)
    layers = [layer1, dropout, layer2]
    optimizer = AdamOptimizer(0.001, layers, 0.9, 0.999)

    model = Model(layers, optimizer, SoftmaxCrossEntropyLoss)
    model.initialize()
    print('Initialized!')

    epochs = 1000
    count = 0
    for i in range(epochs):
        minibatches = random_mini_batches(X_train, Y_train, 128)
        for mini_X, mini_Y in minibatches:
            loss = model.train(mini_X, mini_Y)
            count += 1
            if count % 100 == 1:
                print(loss)

    Y_hat_train = model.predict(X_train)
    Y_train_label = np.argmax(Y_hat_train, axis=0)
    acc = np.sum(Y_train_label.reshape((1, -1)) == Y[:, :m_train]) * 1.0 / m_train
    print("train acc=", acc)

    Y_hat_test = model.predict(X_test)
    Y_test_label = np.argmax(Y_hat_test, axis=0)
    acc = np.sum(Y_test_label.reshape((1, -1)) == Y[:, m_train:]) * 1.0 / (m-m_train)
    print("test acc=", acc)
