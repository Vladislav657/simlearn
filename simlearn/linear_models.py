import numpy as np


class LinearRegressor:
    def __init__(self):
        self.w = None

    def fit(self, X_train, Y_train):
        if len(X_train) != len(Y_train):
            raise ValueError("X_train and Y_train must have the same length")
        if len(X_train) == 0:
            raise ValueError("X_train and Y_train must not be empty")

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        if type(X_train[0]) != np.ndarray:
            X_train = X_train.reshape(-1, 1)

        x_len = len(X_train[0]) + 1
        X = np.array([np.zeros(x_len) for _ in range(x_len)])
        X[0][0] = len(Y_train)

        for i in range(1, x_len):
            X[0][i] = np.sum(X_train[:, i - 1])

        for i in range(1, x_len):
            for j in range(x_len):
                if j < i:
                    X[i][j] = X[j][i]
                else:
                    X[i][j] = np.sum(X_train[:, i - 1] * X_train[:, j - 1])

        Y = np.zeros(x_len)
        Y[0] = np.sum(Y_train)
        for i in range(1, x_len):
            Y[i] = np.sum(Y_train * X_train[:, i - 1])

        self.w = Y @ np.linalg.inv(X)

    def predict(self, X):
        if len(X) == 0:
            raise ValueError("X must not be empty")

        X = np.array(X)
        if type(X[0]) != np.ndarray:
            X = X.reshape(-1, 1)

        return np.array([self.w @ np.array([1, *x]) for x in X])

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
