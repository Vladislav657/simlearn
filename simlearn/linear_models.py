import numpy as np


class LinearRegressor:
    def __init__(self):
        self.w = None
        self.score = None

    def fit(self, X, Y):
        size = len(X)
        n = len(X[0])
        batch_size = min(size, 50)
        alpha = 0.02
        lmd = 0.1
        G = np.ones(n)
        epsilon = 10 ** -8
        self.w = np.ones(n)

        for _ in range(10000):
            k = np.random.randint(0, size - batch_size)
            dL = self._dloss(X[k:k + batch_size], Y[k:k + batch_size])
            G = alpha * G + (1 - alpha) * dL * dL
            self.w -= lmd * dL / (np.sqrt(G) + epsilon)

    def predict(self, X):
        return np.array([self.w @ x.T for x in X])

    def _dloss(self, X, Y):
        return 2 * sum((self.w @ x.T - y) * x for x, y in zip(X, Y)) / len(X) + 0.1 * sum(self.w)

    def _score(self, X, Y):
        return np.mean(self.predict(X) - Y)
