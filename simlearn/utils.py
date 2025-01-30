import numpy as np


def MSE(X, Y, w):
    return np.mean(np.square(Y - X @ w.T))


def dMSE(X, Y, w):
    return -2 / len(X) * X.T @ (Y - X @ w.T)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(X, Y, w):
    return -np.mean(np.array([y * np.log(sigmoid(x @ w.T)) + (1 - y) * np.log(1 - sigmoid(x @ w.T))
                              for x, y in zip(X, Y)]))


def dcross_entropy(X, Y, w):
    return np.mean(np.array([(sigmoid(x @ w.T) - y) * x for x, y in zip(X, Y)]), axis=0)


def is_right(X, Y = None, binary = False):
    if Y is None:
        if len(X) == 0:
            raise ValueError("X must not be empty")
    else:
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length")
        if len(X) == 0:
            raise ValueError("X and Y must not be empty")
        if binary:
            if len(set(Y)) != 2:
                raise ValueError("number of classes must equal 2")


def reduce_data(fit_intercept, X, Y = None):
    if Y is None:
        X = np.array([[1, *x] for x in X]) if fit_intercept else np.array(X)
        if type(X[0]) != np.ndarray:
            X = X.reshape(-1, 1)
        return X
    else:
        X = np.array([[1, *x] for x in X]) if fit_intercept else np.array(X)
        if type(X[0]) != np.ndarray:
            X = X.reshape(-1, 1)
        Y = np.array(Y)
        return X, Y
