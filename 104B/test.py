import numpy as np


def gradient_descent(gradient, start, learn_rate, n_iterations, tolerance):
    vector = start
    for _ in range(n_iterations):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector


# Example usage
# Define the gradient of your function here
def gradient_function(v):
    # Example for a 2D function: f(x, y) = x^2 + y^2
    return np.array([2 * v[0], 2 * v[1]])


# Starting point (can be any point in your domain)
start_point = np.array([10.0, 10.0])

# Learning rate
learn_rate = 0.1

# Number of iterations
n_iterations = 1000

# Tolerance for stopping criterion
tolerance = 1e-6

minimum = gradient_descent(gradient_function, start_point, learn_rate, n_iterations, tolerance)
print("Minimum at:", minimum)


# Multivariable case
def func(v):
    return v[0]**2 - v[1] * v[2]**2 + v[2] * v[3] * v[4]**2


def gradient_function1(v):
    # Example for a 2D function: f(x, y) = x1^2 - x2 * x3^2 + x3 * x4 * x5^2
    return np.array([2 * v[0], -v[1], 2*v[2] + v[3] + v[4], v[2] + v[4], v[3] + v[4]])


start_point = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
minimum = gradient_descent(gradient_function1, start_point, learn_rate, n_iterations, tolerance)
print("Minimum at:", minimum)
ylist = []
for x1 in range(10):
    for x2 in range(10):
        for x3 in range(10):
            for x4 in range(10):
                for x5 in range(10):
                    ylist.append(func([1 + 0.1 * x1, 4 + 0.1 * x2, -6 + 0.1 * x3, 4 + 0.1 * x4, -2 + 0.1 * x5]))
miny = np.min(ylist) - func(minimum)
print(miny)
