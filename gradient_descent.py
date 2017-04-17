import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel(
    "http://college.cengage.com/mathematics/brase/understandable_statistics/8e/students/datasets/slr/excel/slr06.xls")

iter = 0
max_iter = 10000
converged = False
m = len(df.index)
x = np.array(df["X"])
y = np.array(df["Y"])
a = 0.0015
ep = 0.01  # convergence criteria

theta0 = 1
theta1 = 1

J = sum((theta0 + theta1 * x[i] - y[i]) ** 2 for i in range(m))

while not converged:
    grad0 = 1.0 / m * sum([(theta0 + theta1 * x[i] - y[i]) for i in range(m)])
    grad1 = 1.0 / m * sum([(theta0 + theta1 * x[i] - y[i]) * x[i] for i in range(m)])

    temp0 = theta0 - a * grad0
    temp1 = theta1 - a * grad1

    theta0 = temp0
    theta1 = temp1

    e = sum([(theta0 + theta1 * x[i] - y[i]) ** 2 for i in range(m)])

    if abs(J - e) < ep:
        print("Converged, iterations: ", iter, "!!!")
        converged = True
    J = e
    iter += 1
    if iter == max_iter:
        print("Max interactions exceeded!")
        converged = True

plt.scatter(x, y, s=20, c='b', marker="o")
y_predict = []
for i in range(len(x)):
    y_predict.append(theta0 + theta1 * x[i])
plt.plot(x, y_predict)
plt.show()