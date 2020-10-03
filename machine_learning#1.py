from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt


def error(t0, t1, x3, y):
    total_error = 0
    for i in range(0, len(x3)):
        X = x3[i]
        Y = y[i]
        total_error += ((t1 * X + t0) - Y) ** 2

    return total_error / float(len(x3))


def batch(t0, t1, x3, y, learning_rate):
    new_t0 = 0
    new_t1 = 0
    g_t0 = 0
    g_t1 = 0
    M = float(len(x3))

    for i in range(0, int(M)):
        X = x3[i]
        Y = y[i]
        g_t0 += - (1 / M) * ((t0 + (t1 * X)) - Y)
        g_t1 += - (1 / M) * X * ((t0 + (t1 * X)) - Y)

    new_t0 = t0 - (learning_rate * g_t0)
    new_t1 = t1 - (learning_rate * g_t1)
    print(new_t1, new_t0)
    return [new_t0, new_t1]


def gradient_descent(x3, y, t0, t1, learning_rate, iteration):
    for i in range(iteration):
        t0, t1 = batch(t0, t1, x3, y, learning_rate)

    return [t0, t1]


if __name__ == '__main__':
    # y = t0 + t1*x
    gd = read_csv('hw1_data.CSV')
    learning_rate = 0.00001
    t0 = 0
    t1 = 0
    iteration = 400

    x3 = gd['X3 distance to the nearest MRT station'].values.tolist()
    y = gd['Y house price of unit area'].values.tolist()

    [t0, t1] = gradient_descent(x3, y, t0, t1, learning_rate, iteration)
    print(t0, t1)
