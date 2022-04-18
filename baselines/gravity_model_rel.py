# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression

from func import *


def fit(flows, features, train, flight):
    Y = []
    X = []
    for k in flows:
        intensity = k[2]
        feat_o = features[k[0]]
        feat_d = features[k[1]]
        if feat_o[2] == 0:
            feat_o[2] = 1
        if feat_d[3] == 0:
            feat_d[3] = 1
        ft, ff = 1, 1
        if (k[0], k[1]) in train:
            ft = train[(k[0], k[1])]
        if (k[0], k[1]) in flight:
            ff = flight[(k[0], k[1])]

        Y.append(np.log(intensity))
        X.append([np.log(feat_o[2]), np.log(feat_d[3]), np.log(ft), np.log(ff),
                  np.log(haversine_distance(feat_o[0], feat_o[1], feat_d[0], feat_d[1]))])

    reg = LinearRegression().fit(X, Y)
    beta = reg.coef_
    K = np.e**reg.intercept_

    return beta, K


def predict(flows, features, train, flight, beta, K):
    p = []
    r = []
    for f in flows:
        feat_o = features[f[0]][2]
        if feat_o == 0:
            feat_o = 1
        feat_d = features[f[1]][3]
        if feat_d == 0:
            feat_d = 1
        ft, ff = 1, 1
        if (f[0], f[1]) in train:
            ft = train[(f[0], f[1])]
        if (f[0], f[1]) in flight:
            ff = flight[(f[0], f[1])]

        p.append(K * (feat_o ** beta[0]) * (feat_d ** beta[1]) * (ft ** beta[2]) * (ff ** beta[3]) *
                 (haversine_distance(features[f[0]][0], features[f[0]][1], features[f[1]][0], features[f[1]][1]) ** beta[4]))
        r.append(f[2])

    return p, r


if __name__ == '__main__':
    path = '../data/'
    train_f = read_flows([path + 'train.txt', path + 'valid.txt'])  # Flow list: [origin, destination, intensity]
    test_f = read_flows([path + 'test.txt'])
    features = read_features(path + 'features_pop.txt')  # Feature dict: [x/lng, y/lat, push/outgoing, pull/incoming]
    train = load_relation_data(path + 'trains_2017_c281.csv')
    flight = load_relation_data(path + 'flights_201505-201605_c281.csv')

    beta, K = fit(train_f, features, train, flight)
    pred, real = predict(test_f, features, train, flight, beta, K)

    print('beta =', beta, ', K =', K)
    result = np.concatenate((np.array(real).reshape((-1, 1)), np.array(pred).reshape((-1, 1))), axis=1)
    np.savetxt('pred_gm-rel.txt', result, delimiter=',')
    evaluate(pred, real)
