import numpy as np
from simlearn import LinearRegressor
import matplotlib.pyplot as plt

# def func(x):
#     return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5
#
# coord_x = np.arange(-10.0, 10.0, 0.1) # значения по оси абсцисс
# coord_y = func(coord_x) # значения по оси ординат (значения функции)
#
# n = 60
#
# x_train = np.array([[1, x, x ** 2, x ** 3] for x in coord_x])[:-n]
# y_train = coord_y[:-n]
#
# x_test = np.array([[1, x, x ** 2, x ** 3] for x in coord_x])[-n:]
# y_test = coord_y[-n:]
#
# r = LinearRegressor()
# r.fit(x_train, y_train)
# pred_train = r.predict(x_train)
# pred_test = r.predict(x_test)
#
# plt.plot(coord_x, coord_y)
# plt.plot(coord_x[:-n], pred_train)
# plt.plot(coord_x[-n:], pred_test)
# plt.show()
#
#
# print(y_test)
# print(pred_test)
# print(r.w)
import csv


def get_data(filename):
    x_data, y_data = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in list(reader)[1:]:
            x_data.append([float(x) for x in row[:-1]])
            y_data.append(float(row[-1]))
    return np.array(x_data), np.array(y_data)


n = 300

X, Y = get_data("house_price_regression_dataset.csv")
x_train = X[:-n]
y_train = Y[:-n]

x_test = X[-n:]
y_test = Y[-n:]

regressor = LinearRegressor()
regressor.fit(x_train, y_train)
pred = regressor.predict(x_test)

print("real:", end='\t\t')
print(*[round(y, 2) for y in y_test], sep='\t')
print("predicted:", end='\t')
print(*[round(p, 2) for p in pred], sep='\t')

print(np.mean(np.abs(y_test - pred)))
