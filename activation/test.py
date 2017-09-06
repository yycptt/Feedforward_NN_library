import numpy as np

from sigmoid_activation import SigmoidActivation
from relu_activation import ReluActivation
from leakyrelu_activation import LeakyReluActivation
from tanh_activation import TanhActivation

np.random.seed(1)
z = np.array(np.random.randn(2, 2), dtype=np.float64)
da = np.array(np.random.randn(2, 2), dtype=np.float64)
# z = [[ 1.62434542 -0.61175638]
#      [-0.52817178 -1.0729686 ]]
# da = [[ 0.86540765 -2.30153871]
#       [ 1.74481177 -0.76120692]]

sigmoid = SigmoidActivation()
sigmoid_a = sigmoid.forward(z)
sigmoid_dz = sigmoid.backward(da)
assert(np.allclose(sigmoid_a,
                   [[0.83539361, 0.35165864], [0.3709434, 0.25483894]]))
assert(np.allclose(sigmoid_dz,
                   da * (1.0 / (1 + np.exp(-(z+1e-9)))-1.0 / (1 + np.exp(-(z-1e-9)))) / 2e-9))
                   #[[0.11900318, -0.52473897], [0.40714201, -0.14455019]]))
print('Sigmoid Test Passed!')

relu = ReluActivation()
relu_a = relu.forward(z)
relu_dz = relu.backward(da)
assert(np.allclose(relu_a,
                   [[1.62434542, 0], [0, 0]]))
assert(np.allclose(relu_dz,
                   [[0.86540765, 0], [0, 0]]))
print('Relu Test Passed!')

leakyrelu = LeakyReluActivation(0.5)
leakyrelu_a = leakyrelu.forward(z)
leakyrelu_dz = leakyrelu.backward(da)
assert(np.allclose(leakyrelu_a,
                   [[1.62434542, -0.61175638/2], [-0.52817178/2, -1.0729686/2]]))
assert(np.allclose(leakyrelu_dz,
                   [[0.86540765, -2.30153871/2], [1.74481177/2, -0.76120692/2]]))
print('LeakyRelu Test Passed!')

tanh = TanhActivation()
tanh_a = tanh.forward(z)
tanh_dz = tanh.backward(da)
assert(np.allclose(tanh_a,
                   [[0.92525208, -0.54536229], [-0.48398235, -0.79057705]]))
assert(np.allclose(tanh_dz,
                   da * (np.tanh(z+1e-9)-np.tanh(z-1e-9)) / 2e-9))
                   #[[0.12453958, -1.61701512], [1.3361088, -0.28544345]]))
print('Tanh Test Passed!')

print('All Tests Passed!')
