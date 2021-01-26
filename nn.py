import numpy as np
from functions import relu, relu_derivative, tanh, tanh_derivative, cross_entropy, softmax


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, X):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = np.random.rand(in_dim, out_dim)
        self.bias = np.random.rand(1, out_dim)

    def forward(self, X):
        self.input = X
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, out_error, learning_rate=0.05):
        in_error = np.dot(out_error, self.weights.T)
        weights_error = np.dot(self.input.T, out_error)
        bias_error = out_error

        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * bias_error

        return in_error


class Activation(Layer):
    def __init__(self, activation):
        activation_map = {
            'relu': {
                'func': relu,
                'derivative': relu_derivative
            },
            'tanh': {
                'func': tanh,
                'derivative': tanh_derivative
            }
        }
        self.activation = activation_map[activation]['func']
        self.derivative = activation_map[activation]['derivative']

    def forward(self, X):
        self.input = X
        self.output = self.activation(self.input)
        return self.output

    def backward(self, out_error, learning_rate):
        return self.derivative(self.input) * out_error


class Loss(Layer):
    def __init__(self):
        super().__init__()
        self.target = None


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()
        self.softmax_out = None

    def forward(self, X, target):
        self.input = X
        self.target = target
        self.softmax_out = softmax(self.input)
        self.output = cross_entropy(self.softmax_out)
        return self.output

    def backward(self):
        return (1/self.target.shape[0])*(self.softmax_out - self.target)


class MSELoss(Loss):
    def forward(self, X, target):
        self.input = X
        self.target = target
        self.output = np.mean(np.power(target-X, 2))
        return self.output

    def backward(self):
        return (2 * (self.input-self.target)) / self.target.shape[0]


class Model:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def train(self, X, y, epochs=10, learning_rate=0.05):
        for i in range(epochs):
            predictions = self.predict(X)

            loss = self.loss()
            cost = loss.forward(predictions, y)

            error = loss.backward()

            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate)

            print(f'epoch={i}, loss={cost}')

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
