from activation.relu_activation import ReluActivation
from activation.leakyrelu_activation import LeakyReluActivation
from activation.sigmoid_activation import SigmoidActivation
from activation.tanh_activation import TanhActivation
from layer.fully_connected_layer import FullyConnectedLayer
from loss.binary_loss import BinaryLoss
from optimizer.gradient_descent_optimizer import GradientDescentOptimizer
from optimizer.adam_optimizer import AdamOptimizer
from model import Model

import numpy as np
from sklearn.datasets import load_breast_cancer

if __name__ == '__main__':
    data = load_breast_cancer()
    m = data.data.shape[0]
    X = data.data.T
    Y = np.reshape(data.target, (1, m))
    print(X.shape)
    print(Y.shape)

    #layer1 = FullyConnectedLayer(30, 10, ReluActivation(), 0.001)
    layer1 = FullyConnectedLayer(30, 10, TanhActivation(), 0.001)
    #layer1 = FullyConnectedLayer(30, 10, LeakyReluActivation(0.001), 0.001)
    layer2 = FullyConnectedLayer(10, 1, SigmoidActivation(), 0.001)
    layers = [layer1, layer2]
    #optimizer = GradientDescentOptimizer(0.0001, layers)
    optimizer = AdamOptimizer(0.001, layers, 0.9, 0.999)
    model = Model(layers, optimizer, BinaryLoss)
    model.initialize()
    print('Initialized!')

    round = 10000
    losses = []
    for i in range(round):
        loss = model.train(X, Y)
        if i % 10 == 1:
            losses.append(loss)
        if i % 100 == 1:
            print(loss)
    Y_hat = model.predict(X)
    Y_hat[Y_hat >= 0.5] = 1
    Y_hat[Y_hat < 0.5] = 0
    acc = np.sum(Y_hat == Y) * 1.0 / m
    print acc
