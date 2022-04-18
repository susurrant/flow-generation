# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score


def read_flows(filenames):
    data = []
    for fn in filenames:
        with open(fn, 'r') as f:
            flows = f.readlines()
            for flow in flows:
                sl = flow.strip().split('\t')
                data.append([sl[0], sl[1], float(sl[2])])

    return data


def read_features(feature_file, normalized=False):
    features = {}
    min_pull, min_push = float('inf'), float('inf')
    max_pull, max_push = 0, 0

    with open(feature_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sl = line.strip().split('\t')
            features[sl[0]] = list(map(float, sl[1:]))   # [x/lng, y/lat, push, pull]
            if features[sl[0]][2] < min_push:
                min_push = features[sl[0]][2]
            if features[sl[0]][2] > max_push:
                max_push = features[sl[0]][2]
            if features[sl[0]][3] < min_pull:
                min_pull = features[sl[0]][3]
            if features[sl[0]][3] > max_pull:
                max_pull = features[sl[0]][3]
    if normalized:
        for k in features:
            features[k][2] = (features[k][2] - min_push) / (max_push - min_push)
            features[k][3] = (features[k][3] - min_pull) / (max_pull - min_pull)

    return features


def load_relation_data(filename):
    rel = {}
    with open(filename, 'r') as f:
        for line in f:
            sl = line.strip().split('\t')
            rel[(sl[0], sl[1])] = float(sl[2])

    return rel


def evaluate(p, r):
    print('\nnum of test flows:', len(r))
    print('real_min:', min(r), ', real_max:', max(r))
    print('pred_min:', min(p), ', pred_max:', max(p))
    print('real:', r[0:20])
    print('pred:', p[0:20])

    p = np.array(p)
    r = np.array(r)

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
    mape = mape / c1
    print('MAPE:', round(mape, 3))

    rmse = np.sqrt(np.mean(np.square(r - p)))
    print('RMSE:', round(rmse, 3))

    stack = np.column_stack((p, r))
    cpc = 2 * np.sum(np.min(stack, axis=1)) / np.sum(stack)
    print('CPC:', round(cpc, 3))

    smc = stats.spearmanr(r, p)
    print('SMC: correlation =', round(smc[0], 3), ', p-value =', round(smc[1], 3))

    r2 = r2_score(r, p)
    print('R2:', round(r2, 3))

    return [rmse, mape, cpc, smc[0], r2]


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
