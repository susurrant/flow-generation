# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression

from func import *


def fit(flows, train, flight, distance):
    y = []
    X = []
    for k in flows:
        y.append(np.log(k[2]))
        t = np.log(train[(k[0], k[1])]) if (k[0], k[1]) in train else 0
        f = np.log(flight[(k[0], k[1])]) if (k[0], k[1]) in flight else 0
        d = np.log(distance[(k[0], k[1])]) if (k[0], k[1]) in distance else 0
        X.append([t, f, d])

    reg = LinearRegression().fit(X, y)
    beta = reg.coef_
    K = np.e**reg.intercept_

    return beta, K


def predict(flows, train, flight, distance, beta, K):
    p = []
    r = []
    for k in flows:
        t = train[(k[0], k[1])] if (k[0], k[1]) in train else 1
        f = flight[(k[0], k[1])] if (k[0], k[1]) in flight else 1
        d = distance[(k[0], k[1])] if (k[0], k[1]) in distance else 1

        p.append(K * (t ** beta[0]) * (f ** beta[1]) * (d ** beta[2]))
        r.append(k[2])

    return p, r


if __name__ == '__main__':
    path = '../data/'
    train_f = read_flows([path + 'train.txt', path + 'valid.txt'])  # Flow list: [origin, destination, intensity]
    test_f = read_flows([path + 'test.txt'])
    train = load_relation_data(path + 'trains_2017_c281.csv')
    distance = load_relation_data(path + 'distance_c281.csv')
    flight = load_relation_data(path + 'flights_201505-201605_c281.csv')
    print(train_f[:10])
    beta, K = fit(train_f, train, flight, distance)
    pred, real = predict(test_f, train, flight, distance, beta, K)

    print('beta =', beta, ', K =', K)
    result = np.concatenate((np.array(real).reshape((-1, 1)), np.array(pred).reshape((-1, 1))), axis=1)
    np.savetxt('pred_nr.txt', result, delimiter=',')
    evaluate(pred, real)
