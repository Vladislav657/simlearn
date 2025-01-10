import numpy as np


class LinearRegressor:
    def __init__(self, fit_intercept=True):
        self.w = None
        self.fit_intercept = fit_intercept

    def fit(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("X_train and Y_train must have the same length")
        if len(X) == 0:
            raise ValueError("X_train and Y_train must not be empty")

        X = np.array([[1, *x] for x in X]) if self.fit_intercept else np.array(X)
        if type(X[0]) != np.ndarray:
            X = X.reshape(-1, 1)
        Y = np.array(Y)

        self.w = np.linalg.inv(X.T @ X) @ X.T @ Y

    def predict(self, X):
        if len(X) == 0:
            raise ValueError("X must not be empty")

        X = np.array([[1, *x] for x in X]) if self.fit_intercept else np.array(X)
        if type(X[0]) != np.ndarray:
            X = X.reshape(-1, 1)

        return np.array([self.w @ x for x in X])

    def score(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        if len(X) == 0:
            raise ValueError("X and Y must not be empty")

        X = np.array(X)
        Y = np.array(Y)
        if type(X[0]) != np.ndarray:
            X = X.reshape(-1, 1)

        return np.mean(np.square(self.predict(X) - Y))
