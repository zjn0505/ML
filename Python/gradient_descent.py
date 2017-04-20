import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

df = pd.read_excel(
    "http://college.cengage.com/mathematics/brase/understandable_statistics/8e/students/datasets/slr/excel/slr06.xls")

x = np.array(df["X"])
y = np.array(df["Y"])
m = len(df.index)

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0

    theta0 = 1
    theta1 = 1

    J = sum((theta0 + theta1 * x[i] - y[i]) ** 2 for i in range(m))

    while not converged:
        grad0 = 1.0 / m * sum([(theta0 + theta1 * x[i] - y[i]) for i in range(m)])
        grad1 = 1.0 / m * sum([(theta0 + theta1 * x[i] - y[i]) * x[i] for i in range(m)])

        temp0 = theta0 - alpha * grad0
        temp1 = theta1 - alpha * grad1

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
    return theta0, theta1

plt.scatter(x, y, s=20, c='b', marker="o")
y_predict = []
alpha = 0.0015
ep = 0.0001
theta0, theta1 = gradient_descent(alpha, x, y, ep)
for i in range(len(x)):
    y_predict.append(theta0 + theta1 * x[i])
# plt.plot(x, y_predict)

# OR using Matrix Vector Multiplication for calculation
dataMatrix = np.stack((np.ones(m), x), axis=-1)
parameters = [theta0, theta1]
prediction = dataMatrix.dot(parameters)
plt.plot(x, prediction)
plt.show()
