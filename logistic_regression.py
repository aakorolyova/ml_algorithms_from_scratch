import numpy as np
from numpy import log, dot, exp, shape
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def standardise(X):
    X_st = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_st


def f1_score(y_true, y_pred, label=1):
    tp = np.sum((y_true == label) & (y_true == y_pred))
    fp = np.sum((y_true != label) & (y_true != y_pred))
    fn = np.sum((y_true == label) & (y_true != y_pred))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score


class LogisticRegression:

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def sigmoid(self, z):
        sig = 1 / (1 + exp(-z))
        return sig

    def initialize(self, X):
        weights = np.zeros((shape(X)[1] + 1, 1))
        X = np.concatenate((np.ones((shape(X)[0], 1)), X), axis=1)
        return weights, X

    def fit(self, X, y):
        weights, X = self.initialize(X)

        def cost(theta):
            pred = self.sigmoid(np.matmul(X, theta))
            cost0 = np.matmul(y, log(pred))
            cost1 = np.matmul(1 - y, log(1 - pred))
            cost = -(cost1 + cost0) / len(y)
            return cost

        cost_list = np.zeros(self.iterations)

        for i in range(self.iterations):
            pred = self.sigmoid(np.matmul(X, weights))
            weights -= self.learning_rate * np.matmul(X.T, pred - np.reshape(y, (len(y), 1)))
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list

    def predict(self, X):
        pred = self.sigmoid(np.matmul(self.initialize(X)[1], self.weights))
        res = np.where(pred > 0.5, 1, 0).flatten()
        return res


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=5)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

    X_tr = standardise(X_tr)
    X_te = standardise(X_te)
    model = LogisticRegression(learning_rate=0.001, iterations=400)
    model.fit(X_tr, y_tr)

    y_test_pred = model.predict(X_te)
    y_train_pred = model.predict(X_tr)

    # Let's see the f1-score for training and testing data
    f1_score_tr = f1_score(y_tr, y_train_pred)
    f1_score_te = f1_score(y_te, y_test_pred)
    print(f1_score_tr)
    print(f1_score_te)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    model = LogisticRegression().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f1_score(y_te, y_pred))
