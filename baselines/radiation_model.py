# -*- coding: utf-8 -*-
from func import *


def load_flow_data(path):
    train_f = read_flows(path + 'train.txt')
    test_f = read_flows(path + 'test.txt')
    validation_f = read_flows(path + 'valid.txt')

    total_f = {}
    for f in train_f:
        if f[0] not in total_f:
            total_f[f[0]] = 0
        total_f[f[0]] += f[2]

    for f in test_f:
        if f[0] not in total_f:
            total_f[f[0]] = 0
        total_f[f[0]] += f[2]

    for f in validation_f:
        if f[0] not in total_f:
            total_f[f[0]] = 0
        total_f[f[0]] += f[2]

    return test_f, total_f


def predict(test_f, total_f, features):
    real = []
    pred = []
    for f in test_f:
        real.append(f[2])
        ogid = f[0]
        dgid = f[1]
        # Feature dict: [x/lng, y/lat, feature]
        opop = features[ogid][2]
        dpop = features[dgid][2]
        d = haversine_distance(features[ogid][0], features[ogid][1], features[dgid][0], features[dgid][1])

        s = 0
        for gid in features:
            if gid != ogid and gid != dgid:
                if haversine_distance(features[ogid][0], features[ogid][1], features[gid][0], features[gid][1]) <= d:
                    s += features[gid][2]
        pred.append(total_f[ogid]*opop*dpop/((opop+s)*(opop+dpop+s)))

    return pred, real


if __name__ == '__main__':
    path = '../data/simple/'
    test_flows, total_flows = load_flow_data(path)
    features = read_features(path + 'features.txt')

    pred, real = predict(test_flows, total_flows, features)
    #np.savetxt('../data/pred_RM.txt', pred, delimiter=',')
    evaluate(pred, real)
