import numpy as np

class NaiveBayes:
    def __init__(self):
        self.num_classes = 0
        self.num_features = 0
        self.log_prior = None
        self.log_likelihood = None
    
    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.num_features = X.shape[1]
        self.log_prior = np.zeros(self.num_classes)
        self.log_likelihood = np.zeros((self.num_features, self.num_classes))

        for c in range(self.num_classes):
            self.log_prior[c] = np.log(len(y[y == c]) / len(y))
        
        for f in range(self.num_features):
            for c in range(self.num_classes):
                sub_x = X[y == c]
                self.log_likelihood[f, c] = np.log((np.sum(sub_x[:, f]) + 1) / (np.sum(sub_x)) + self.num_features)

        return self

    def predict(self, X):
        probs = np.dot(X, self.log_likelihood) + self.log_prior.reshape((1, -1))
        return np.argmax(probs, axis=1)