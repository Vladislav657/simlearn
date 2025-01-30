from simlearn import LinearRegressor, SGDRegressor
import csv


def get_data(filename):
    x_data, y_data = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in list(reader)[1:]:
            x_data.append([float(x) for x in row[:-1]])
            y_data.append(float(row[-1]))
    return x_data, y_data


n = 300

X, Y = get_data("house_price_regression_dataset.csv")
x_train = X[:-n]
y_train = Y[:-n]

x_test = X[-n:]
y_test = Y[-n:]

SGD_regressor = SGDRegressor(eta=0.01, n_iter=10000, batch_size=40)
SGD_regressor.fit(x_train, y_train)
SGD_pred = SGD_regressor.predict(x_test)

linear_regressor = LinearRegressor()
linear_regressor.fit(x_train, y_train)
linear_pred = linear_regressor.predict(x_test)

print("real:", end='\t\t')
print(*[round(y, 2) for y in y_test], sep='\t')
print("linear:", end='\t\t')
print(*[round(p, 2) for p in linear_pred], sep='\t')
print("SGD:", end='\t\t')
print(*[round(p, 2) for p in SGD_pred], sep='\t')

# print(regressor.predict([[10000,10,30,2025,7.11,40,0]]))
# print(np.mean(np.abs(y_test - pred)))
# print(np.array([10000,10,30,2025,7.11,40,0]).reshape(-1, 1))
# print(regressor.score(x_test, y_test))
