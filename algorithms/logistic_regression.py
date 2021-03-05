import numpy as np
from sklearn.preprocessing import OneHotEncoder

class LogisticRegressor:
    def __init__(self, num_iter=500, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.num_classes = 0
        self.num_features = 0
        self.num_samples = 0
        self.num_iter = num_iter
    
    def fit(self, X, y):
        encoder = OneHotEncoder(sparse=False)
        labels = encoder.fit_transform(y.reshape(-1, 1))
        self.num_classes = len(encoder.categories_[0])
        self.num_samples, self.num_features = X.shape
        self.weights = np.zeros((self.num_classes, self.num_features))
        self.bias = np.random.rand(self.num_classes, 1)

        for i in range(self.num_iter):
            probs = self.predict_proba(X)
            loss = -(1/self.num_samples) * np.sum(np.log(probs * labels, where=labels.astype(bool)))
            print(f'iter={i}, loss={loss}')
            diff = np.transpose(probs-labels)
            grad_weights = (1/self.num_samples) * np.dot(diff, X)
            grad_bias = (1/self.num_samples) * np.dot(diff, np.ones((self.num_samples, 1)))
            self.weights = self.weights - self.learning_rate * grad_weights
            self.bias = self.bias - self.learning_rate * grad_bias
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights.T) + self.bias.reshape((1, -1))
        z_exp = np.exp(z)
        z_exp_sum = np.sum(z_exp, axis=1, keepdims=True)
        probs = z_exp / z_exp_sum
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)