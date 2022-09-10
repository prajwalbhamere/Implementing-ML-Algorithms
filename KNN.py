import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

def EuclideanDistance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def accuracy(y_test, y_pred):
    return np.sum(y_pred == y_test)/len(y_test)

class KNN:

    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y):
        self. X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # calculate distances
        distances = [EuclideanDistance(x, x_train) for x_train in self.X_train]

        # k nearest samples
        k_index = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_index]

        # most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
clf = KNN(k = 5)  # usually odd
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy(y_test, pred)

print("The accuracy of KNN algorithm is : ", accuracy)