from audioop import bias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise = 20, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)

# plt.scatter(X[:, 0], y)
# plt.show()

class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape     # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias         # predicted y (feature)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))       # derivative of weight
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw  # updating values of weights and bias according to the formulas
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

def MSE(y_true, y_pred):                           # Mean Squared Error
    accuracy = np.mean((y_true - y_pred) ** 2)
    return accuracy

reg = LinearRegression(lr = 0.01)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

accuracy = MSE(y_test, y_pred)
print("The accuracy of Linear Regression algorithm is : ", accuracy)
