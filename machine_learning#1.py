from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import numpy as np



def graph(b, X3, x3, y):
    pred = X3 * b[1] + b[0]
    pred = np.array(pred).reshape(414, 1)

    plt.scatter(x3,y)
    plt.scatter(x3, pred)

    plt.title('gradient descent #1', fontsize=13)
    plt.xlabel('x3')
    plt.ylabel('y')
    plt.savefig('100000.png')

    plt.show()
    return pred

def normalize(x3, y):
    for i in range(len(x3)):
        x3[i] = (x3[i]/6488.021)   # 정규화
        y[i] = (y[i]/117.5)
    return x3, y



def gradient_descent(X, y, theta, learning_rate, iteration):


    m = len(y)
    for it in range(iteration):

        theta = theta - learning_rate*(X.dot(X.T.dot(theta)-y)/m)

    return theta

def matrix(x3, y):
    th1 = [1 for i in range(414)]  # 1로 이루어진

    X = []
    X.append(th1)
    X.append(x3)
    X = np.asmatrix(X)  # 2*414 1과 x3값으로 이루어짐

    Y = np.array(y).reshape(414, 1)
    return X, Y




if __name__ == '__main__':
    gd = read_csv('hw1_data.CSV')

    learning_rate = 0.0001
    iteration = 100000
    theta = np.array([0.633, -0.09])  # 초기 설정값
    theta = np.array(theta).reshape(2, 1)

    th1 = [1 for i in range(414)]

    x3 = ((gd['X3 distance to the nearest MRT station']).values.tolist())
    y = ((gd['Y house price of unit area']).values.tolist())

    x3 , y = normalize(x3, y)

    X3 = np.array(x3).reshape(414, 1)  # x3 array
    x3 = np.array(x3)
    y = np.array(y)  # y array

    X , Y = matrix(x3, y , th1)

    b = gradient_descent(X, Y,  theta, learning_rate, iteration)
    pred = graph(b, X3, x3, y)




