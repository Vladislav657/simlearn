import numpy as np


class LinearRegressor:
    def __init__(self, fit_intercept=True):
        self.w = None
        self.fit_intercept = fit_intercept

    def fit(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        if len(X) == 0:
            raise ValueError("X and Y must not be empty")

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


class LinearBinaryClassifier:
    def __init__(self, fit_intercept=True):
        self.w = None
        self.fit_intercept = fit_intercept

    def fit(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        if len(X) == 0:
            raise ValueError("X and Y must not be empty")
        if len(set(Y)) != 2:
            raise ValueError("number of classes must equal 2")

        X = np.array([[1, *x] for x in X]) if self.fit_intercept else np.array(X)
        if type(X[0]) != np.ndarray:
            X = X.reshape(-1, 1)
        Y = np.array(Y)

        pt = np.sum([x * y for x, y in zip(X, Y)], axis=0)
        xt = np.sum([np.outer(x, x) for x in X], axis=0)
        self.w = np.dot(pt, np.linalg.inv(xt))

    def predict(self, X):
        if len(X) == 0:
            raise ValueError("X must not be empty")

        X = np.array([[1, *x] for x in X]) if self.fit_intercept else np.array(X)
        if type(X[0]) != np.ndarray:
            X = X.reshape(-1, 1)

        return np.array([np.sign(self.w @ x) for x in X], dtype=np.int32)

    def score(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        if len(X) == 0:
            raise ValueError("X and Y must not be empty")
        if len(set(Y)) != 2:
            raise ValueError("number of classes must equal 2")

        X = np.array([[1, *x] for x in X]) if self.fit_intercept else np.array(X)
        Y = np.array(Y)
        if type(X[0]) != np.ndarray:
            X = X.reshape(-1, 1)

        return np.mean([(1 - self.w.T @ x * y) ** 2 for x, y in zip(X, Y)])
