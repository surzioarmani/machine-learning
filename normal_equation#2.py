from pandas.io.parsers import read_csv
import numpy as np
import numpy.linalg as lin


def Standard(x3, y):

    for i in range(len(x3)):
        x3[i] = (x3[i]/6488.021)   # 정규화
        y[i] = (y[i]/117.5)
    return x3, y


def Matrix(x3, y):

    th1 = [1 for i in range(414)]
    X = []
    X.append(th1)
    X.append(x3)
    X = np.asmatrix(X).T  # 2*414 1과 x3값으로 이루어짐
    Y = np.array(y).reshape(414, 1)
    return X, Y


def Normal_equation(X, Y):

    Theta = lin.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return Theta


if __name__ == '__main__':
    ne = read_csv('hw1_data.CSV')

    x3 = ((ne['X3 distance to the nearest MRT station']).values.tolist())
    y = ((ne['Y house price of unit area']).values.tolist())

    x3 , y = Standard(x3, y)          # 0 <= x3, y <= 1
    X3 = np.array(x3).reshape(414, 1)  # x3 array
    y = np.array(y)                    # y array
    X , Y = Matrix(x3, y)              # matrix로 변환

    Theta = Normal_equation(X, Y)  #gradient descent

    print('Theta0: ' + str(Theta[0]) + '\n' +
          'Theta2: ' + str(Theta[1]) + '\n')




