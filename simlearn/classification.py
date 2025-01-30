import numpy as np
from .utils import is_right, reduce_data, sigmoid, dcross_entropy, cross_entropy


class LinearBinaryClassifier:
    def __init__(self, fit_intercept=True):
        self.w = None
        self.fit_intercept = fit_intercept

    def fit(self, X, Y):
        is_right(X, Y, binary=True)
        X, Y = reduce_data(self.fit_intercept, X, Y)
        pt = np.sum([x * y for x, y in zip(X, Y)], axis=0)
        xt = np.sum([np.outer(x, x) for x in X], axis=0)
        self.w = np.dot(pt, np.linalg.inv(xt))

    def predict(self, X):
        is_right(X)
        X = reduce_data(self.fit_intercept, X)
        return np.array([np.sign(self.w @ x) for x in X], dtype=np.int32)

    def score(self, X, Y):
        is_right(X, Y, binary=True)
        X, Y = reduce_data(self.fit_intercept, X, Y)
        return np.mean([(1 - self.w.T @ x * y) ** 2 for x, y in zip(X, Y)])


class LogisticRegression:
    def __init__(self, eta=0.01, n_iter=1000, batch_size=50, fit_intercept=True):
        self.w = None
        self.eta = eta
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.xm = None
        self.xd = None

    def fit(self, X, Y):
        is_right(X, Y, binary=True)
        X, Y = reduce_data(self.fit_intercept, X, Y)
        Y = Y - np.min(Y)
        Y = Y / np.max(Y)

        self.xm = np.mean(X, axis=0)
        self.xd = np.std(X, axis=0)
        X = (X - self.xm) / (self.xd + 1e-10)
        self.w = np.random.randn(len(X[0]))

        N = len(Y)
        batch_size = min(N, self.batch_size)
        for i in range(self.n_iter):
            k = np.random.randint(0, N - batch_size + 1)
            self.w -= self.eta * dcross_entropy(X[k: k + batch_size], Y[k: k + batch_size], self.w)

    def predict(self, X):
        is_right(X)
        X = reduce_data(self.fit_intercept, X)
        X = (X - self.xm) / (self.xd + 1e-10)
        return np.round(sigmoid(X @ self.w.T))

    def score(self, X, Y):
        is_right(X, Y, binary=True)
        X, Y = reduce_data(self.fit_intercept, X, Y)
        X = (X - self.xm) / (self.xd + 1e-10)
        Y = Y - np.min(Y)
        Y = Y / np.max(Y)
        return cross_entropy(X, Y, self.w)


class SoftmaxClassifier:
    pass
