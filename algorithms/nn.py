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

    def __repr__(self):
        return f'Linear_{self.in_dim}_{self.out_dim}'

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

    def __repr__(self):
        return f'Activation_{self.activation_name}'

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
        self.output = cross_entropy(self.softmax_out, self.target)
        return self.output

    def backward(self):
        return (1/self.target.shape[0])*(self.softmax_out - self.target)

    def __repr__(self):
        return 'CrossEntropyLoss'


class MSELoss(Loss):
    def forward(self, X, target):
        self.input = X
        self.target = target
        self.output = np.mean(np.power(target-X, 2))
        return self.output
    
    def backward(self):
        return (2 * (self.input-self.target)) / self.target.shape[0]

    def __repr__(self):
        return 'MSELoss'


class Model:
    def __init__(self, name):
        self.name = name
        self.layers = []
        self.loss = None
    
    def add(self, layer):
        self.layers.append(layer)
    
    def set_loss(self, loss):
        self.loss = loss
    
    def train(self, data_generator, epochs=100, learning_rate=0.05, checkpoint=100):
        for i in range(epochs):
            data = data_generator()
            epoch_loss = 0

            for X, y in data:
                predictions = self.predict(X)
                
                loss = self.loss()
                cost = loss.forward(predictions, y)
                
                error = loss.backward()
                
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

                epoch_loss += cost
                
            print(f'epoch={i}, loss={epoch_loss}')

            if i % checkpoint == 0:
                self.save()
    
    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def save(self):
        now = datetime.now()
        id = now.strftime('%Y%m%d%H%M%S')
        with open(f'./models/{self.name}-{id}.pkl', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(model_path):
        with open(f'./{model_path}', 'rb') as f:
            return pickle.load(f)

    def summary(self):
        print(f'Model {self.name}')
        for layer in self.layers:
            print(layer)