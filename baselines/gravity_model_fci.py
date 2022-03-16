# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression

from func import *


def read_fci(filename):
    fci = {}
    with open(filename, 'r') as f:
       line = f.readline().strip()
       while line:
           s = line.split('\t')
           fci[(s[0], s[1])] = float(s[2])
           line = f.readline().strip()

    return fci


def fit(flows, features, fci):
    Y = []
    X = []
    for k in flows:
        intensity = k[2]
        feat_o = features[k[0]]
        feat_d = features[k[1]]
        Y.append(np.log(intensity))
        X.append([np.log(feat_o[2]), np.log(feat_d[2]), np.log(fci[(k[0], k[1])]),
                  np.log(haversine_distance(feat_o[0], feat_o[1], feat_d[0], feat_d[1]))])

    reg = LinearRegression().fit(X, Y)
    beta = reg.coef_
    K = np.e**reg.intercept_

    return beta, K


def predict(flows, features, fci, beta, K):
    p = []
    r = []
    for f in flows:
        p.append(K * (features[f[0]][2] ** beta[0]) * (features[f[1]][2] ** beta[1]) * (fci[(f[0], f[1])] ** beta[2]) *
                 (haversine_distance(features[f[0]][0], features[f[0]][1], features[f[1]][0], features[f[1]][1]) ** beta[3]))
        r.append(f[2])

    return p, r


if __name__ == '__main__':
    path = '../data/flow/'
    train_f = read_flows(path + 'train.txt')  # Flow list: [origin, destination, intensity]
    test_f = read_flows(path + 'test.txt')
    features = read_features(path + 'features.txt')  # Feature dict: [x/lng, y/lat, feature]
    fci = read_fci(path + 'fci.txt')

    beta, K = fit(train_f, features, fci)
    pred, real = predict(test_f, features, fci, beta, K)

    print('beta =', beta, ', K =', K)
    # np.savetxt('../data/pred_GM_P.txt', pred, delimiter=',')
    evaluate(pred, real)
