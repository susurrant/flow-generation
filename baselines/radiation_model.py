# -*- coding: utf-8 -*-
from func import *


def predict(test_f, total_f, features):
    real = []
    pred = []
    for f in test_f:
        real.append(f[2])
        ogid = f[0]
        dgid = f[1]
        # Feature dict: [x/lng, y/lat, push, pull]
        opop = features[ogid][2]
        dpop = features[dgid][3]
        d = haversine_distance(features[ogid][0], features[ogid][1], features[dgid][0], features[dgid][1])

        s = 0
        for gid in features:
            if gid != ogid and gid != dgid:
                if haversine_distance(features[ogid][0], features[ogid][1], features[gid][0], features[gid][1]) < d:
                    s += features[gid][2]

        r = total_f[ogid][2] * opop * dpop / ((opop + s) * (opop + dpop + s))
        pred.append(r)
    # print(min(pred), max(pred))
    return pred, real


if __name__ == '__main__':
    path = '../data/'
    test_flows = read_flows([path + 'test.txt'])
    population = read_features(path + 'features_pop.txt')
    total_f = read_features(path + 'features_flow.txt')

    pred, real = predict(test_flows, total_f, population)
    result = np.concatenate((np.array(real).reshape((-1, 1)), np.array(pred).reshape((-1,1))), axis=1)
    np.savetxt('pred_RM.txt', result, delimiter=',')
    evaluate(pred, real)
