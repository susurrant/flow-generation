# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy import stats


def read_flows(filename):
    data = []
    with open(filename, 'r') as f:
        flows = f.readlines()
        for flow in flows:
            sl = flow.strip().split('\t')
            data.append([sl[0], sl[1], float(sl[2])])

    return data


def read_features(feature_file):
    features = {}
    with open(feature_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sl = line.strip().split('\t')
            features[sl[0]] = list(map(float, sl[1:]))
    return features


def evaluate(p, r):
    print('\nnum of test flows:', len(r))
    print('real_min:', min(r), ', real_max:', max(r))
    print('pred_min:', min(p), ', pred_max:', max(p))
    print('real:', r[0:20])
    print('pred:', p[0:20])

    p = np.array(p)
    r = np.array(r)

    #print('MAE:', round(np.mean(np.abs(r - p)),3))

    c1 = 0
    mape = 0
    c2 = 0
    ssi = 0
    for i in range(p.shape[0]):
        if r[i]:
            mape += np.abs((r[i] - p[i]) / r[i])
            c1 += 1
        if r[i] + p[i]:
            ssi += min(r[i], p[i]) / (r[i] + p[i])
            c2 += 1
    print('MAPE:', round(mape / c1, 3))

    #print('MSE:', round(np.mean(np.square(r - p)), 3))
    print('RMSE:', round(np.sqrt(np.mean(np.square(r - p))), 3))

    stack = np.column_stack((p, r))
    print('CPC:', round(2 * np.sum(np.min(stack, axis=1)) / np.sum(stack), 3))

    #print('SSI:', round(ssi * 2 / (c2 ^ 2), 3))

    smc = stats.spearmanr(r, p)
    print('SMC: correlation =', round(smc[0], 3), ', p-value =', round(smc[1], 3))

    #llr = stats.linregress(r, p)
    #print('LLR: R =', round(llr[2], 3), ', p-value =', round(llr[3], 3))


def haversine(theta):
    v = math.sin(theta / 2)
    return v * v


def haversine_distance(lng1, lat1, lng2, lat2, EARTH_RADIUS=6371.0):
    lat1 *= math.pi / 180
    lng1 *= math.pi / 180
    lat2 *= math.pi / 180
    lng2 *= math.pi / 180

    vLng = abs(lng1 - lng2)
    vLat = abs(lat1 - lat2)

    h = haversine(vLat) + math.cos(lat1) * math.cos(lat2) * haversine(vLng)
    distance = 2 * EARTH_RADIUS * math.asin(math.sqrt(h))

    return distance
