import numpy as np
from .utils import is_right, reduce_data, MSE, dMSE


class LinearRegressor:
    def __init__(self, fit_intercept=True):
        self.w = None
        self.fit_intercept = fit_intercept

    def fit(self, X, Y):
        is_right(X, Y)
        X, Y = reduce_data(self.fit_intercept, X, Y)
        self.w = np.linalg.inv(X.T @ X) @ X.T @ Y

    def predict(self, X):
        is_right(X)
        X = reduce_data(self.fit_intercept, X)
        return X @ self.w.T

    def score(self, X, Y):
        is_right(X, Y)
        X, Y = reduce_data(self.fit_intercept, X, Y)
        return MSE(X, Y, self.w)


class SGDRegressor:
    def __init__(self, eta=0.01, n_iter=1000, batch_size=50, fit_intercept=True):
        self.w = None
        self.eta = eta
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.xm, self.ym = None, None
        self.xd, self.yd = None, None

    def fit(self, X, Y):
        is_right(X, Y)
        X, Y = reduce_data(self.fit_intercept, X, Y)

        self.xm, self.ym = np.mean(X, axis=0), np.mean(Y)
        self.xd, self.yd = np.std(X, axis=0), np.std(Y)
        X, Y = (X - self.xm) / (self.xd + 1e-10), (Y - self.ym) / (self.yd + 1e-10)
        self.w = np.random.randn(len(X[0]))

        N = len(Y)
        batch_size = min(N, self.batch_size)
        for i in range(self.n_iter):
            k = np.random.randint(0, N - batch_size + 1)
            self.w -= self.eta * dMSE(X[k: k + batch_size], Y[k: k + batch_size], self.w)

    def predict(self, X):
        is_right(X)
        X = reduce_data(self.fit_intercept, X)
        X = (X - self.xm) / (self.xd + 1e-10)
        return X @ self.w.T * self.yd + self.ym

    def score(self, X, Y):
        is_right(X, Y)
        X, Y = reduce_data(self.fit_intercept, X, Y)
        return MSE(X, Y, self.w)


class RidgeRegressor:
    pass


class LassoRegressor:
    pass
