import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

class LogisticRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):

        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw  # updating weights and bias
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)

        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_cls

    def _sigmoid(self, X):
        return 1/(1 + np.exp(-X))

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

reg = LogisticRegression(lr = 0.0001)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
accuracy = accuracy(y_test, pred)

print("The accuracy of Logistic Regression is : ", accuracy)