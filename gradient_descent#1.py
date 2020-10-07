from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import numpy as np

# 작성자: 조수아 
# 작성일: 2020.10.07
# Machine learning --> gradient_descent/linear regression

def Cost(cost):

    return min(cost)


def Graph(b, X3, x3, y, cost):
    pred = X3 * b[1] + b[0]
    pred = np.array(pred).reshape(414, 1)

    plt.scatter(x3,y)
    plt.plot(x3, pred, color = 'red')

    plt.title('gradient descent #1', fontsize=13)
    plt.xlabel('x3')
    plt.ylabel('y')
    plt.savefig('20000.png')

    plt.show()
    return pred

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
    X = np.asmatrix(X)  # 2*414 1과 x3값으로 이루어짐
    Y = np.array(y).reshape(414, 1)

    return X, Y


def Gradient_descent(X, y, theta, learning_rate, iteration):
    m = len(y)
    cost = []
    for it in range(iteration):
        cost_ = (((X.T.dot(theta) - y)).T.dot((X.T.dot(theta) - y)) / (2 * m))
        cost.append(cost_)
        theta = theta - learning_rate * (X.dot(X.T.dot(theta)-y) / m)

    return theta, cost


if __name__ == '__main__':
    gd = read_csv('hw1_data.CSV')

    learning_rate = 0.01
    iteration = 20000                           # 초기 iteration 임의 설정
    theta = np.array([1, 1]).reshape(2,1)  # 초기 theta0, theta3 임의 설정

    x3 = ((gd['X3 distance to the nearest MRT station']).values.tolist())
    y = ((gd['Y house price of unit area']).values.tolist())

    x3 , y = Standard(x3, y)          # 0 <= x3, y <= 1

    X3 = np.array(x3).reshape(414, 1)  # x3 array
    y = np.array(y)                    # y array

    X , Y = Matrix(x3, y)              # matrix로 변환

    Theta , cost= Gradient_descent(X, Y,  theta, learning_rate, iteration)  #gradient descent
    Graph(Theta, X3, x3, y, cost)  # 그래프로 확인
    J = Cost(cost)   # cost function 값 확인하고 싶을 경우 활성화
    print('Cost function value: ' + str(Cost(cost)))
    print('Theta0: ' + str(Theta[0]) + '\n' +
          'Theta2: ' + str(Theta[1]) + '\n')




