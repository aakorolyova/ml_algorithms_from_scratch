import numpy as np
from numpy import log, dot, exp, shape
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def standardise(X):
    X_st = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_st


def mse(y_true, y_pred):
    return (np.square(y_true - y_pred)).mean(axis=None)


class LinearRegression:

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def initialize(self, X):
        weights = np.zeros((shape(X)[1] + 1, 1))
        X = np.concatenate((np.ones((shape(X)[0], 1)), X), axis=1)
        return weights, X

    def fit(self, X, Y):
        weights, X = self.initialize(X)

        def cost(theta):
            pred = np.matmul(X, theta)
            cost = (np.square(Y - pred)).mean(axis=None)
            return cost

        cost_list = np.zeros(self.iterations)
        for i in range(self.iterations):
            pred = np.matmul(X, weights)
            dW = - (2 * np.matmul(X.T, np.reshape(Y, (len(Y), 1)) - pred)) / len(Y)
            weights -= self.learning_rate * dW
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list

    def predict(self, X):
        return np.matmul(self.initialize(X)[1], self.weights)


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=5)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

    X_tr = standardise(X_tr)
    X_te = standardise(X_te)

    model = LinearRegression(learning_rate=0.005, iterations=1000)
    model.fit(X_tr, y_tr)

    y_test_pred = model.predict(X_te).flatten()
    y_train_pred = model.predict(X_tr).flatten()

    # Let's see the MSE for training and testing data
    mse_tr = mse(y_tr, y_train_pred)
    mse_te = mse(y_te, y_test_pred)
    print(mse_tr)
    print(mse_te)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    model = LinearRegression().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(mean_squared_error(y_te, y_pred))
