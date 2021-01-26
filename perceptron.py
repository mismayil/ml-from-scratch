import numpy as np


def perceptron(x, y):
    theta = np.zeros(x[0].shape)
    theta_0 = 0
    converged = False
    mistakes = [0] * len(x)

    while not converged:
        converged = True

        for i in range(len(x)):
            if y[i] * (np.dot(theta, np.transpose(x[i])) + theta_0) <= 0:
                converged = False
                mistakes[i] += 1
                theta += y[i] * x[i]
                theta_0 += y[i]

    return theta, theta_0, mistakes


x1 = np.array([[-4, 2]])
x2 = np.array([[-2, 1]])
x3 = np.array([[-1, -1]])
x4 = np.array([[2, 2]])
x5 = np.array([[1, -2]])
y1 = 1
y2 = 1
y3 = -1
y4 = -1
y5 = -1
x = [x1, x2, x3, x4, x5]
y = [y1, y2, y3, y4, y5]
theta, theta_0, mistakes = perceptron(x, y)
print(f'theta={theta}, theta_0={theta_0}, mistakes={mistakes}')
