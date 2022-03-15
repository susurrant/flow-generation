# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression

from func import *


def fit(flows, features):
    Y = []
    X = []
    for k in flows:
        intensity = k[2]
        feat_o = features[k[0]]
        feat_d = features[k[1]]
        Y.append(np.log(intensity))
        X.append([np.log(feat_o[2]), np.log(feat_d[2]),
                  np.log(haversine_distance(feat_o[0], feat_o[1], feat_d[0], feat_d[1]))])

    reg = LinearRegression().fit(X, Y)
    beta = reg.coef_
    K = np.e**reg.intercept_

    return beta, K


def predict(flows, features, beta, K):
    p = []
    r = []
    for f in flows:
        p.append(K * (features[f[0]][2] ** beta[0]) * (features[f[1]][2] ** beta[1]) *
                 (haversine_distance(features[f[0]][0], features[f[0]][1], features[f[1]][0], features[f[1]][1]) ** beta[2]))
        r.append(f[2])

    return p, r


if __name__ == '__main__':
    path = '../data/simple/'
    train_f = read_flows(path + 'train.txt')  # Flow list: [origin, destination, intensity]
    test_f = read_flows(path + 'test.txt')
    features = read_features(path + 'features.txt')  # Feature dict: [x/lng, y/lat, feature]

    beta, K = fit(train_f, features)
    pred, real = predict(test_f, features, beta, K)

    print('beta =', beta, ', K =', K)
    # np.savetxt('../data/pred_GM_P.txt', pred, delimiter=',')
    evaluate(pred, real)
