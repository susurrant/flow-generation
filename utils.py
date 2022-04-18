# -*- coding: utf-8 -*-
"""
author: Xin Yao
create date: 2021-12-02
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader


def read_setting(file_name):
    setting = {}
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            s = line.strip().split('=')
            setting[s[0]] = s[1]
    return setting


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def load_entity_relation(file_path):
    with open(os.path.join(file_path, 'entities.dict')) as f:
        entity2id = dict()

        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relations.dict')) as f:
        relation2id = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    return entity2id, relation2id


def load_graph_data(file_path, verbose=1):
    print('\nLoad graph data...')
    entity2id, relation2id = load_entity_relation(file_path)

    graph_triplets = []
    with open(os.path.join(file_path, 'graph.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            graph_triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    if verbose:
        print('\tGraph structure:')
        print('\t\tnum_entity: {}'.format(len(entity2id)))
        print('\t\tnum_relation: {}'.format(len(relation2id)))
        print('\t\tnum_graph_triples: {}'.format(len(graph_triplets)))

    embedding_graph = build_embedding_graph(len(entity2id), len(relation2id), np.array(graph_triplets))

    return entity2id, relation2id, embedding_graph


def load_flow_data(file_path, entity2id, mode, batch_size=0):
    print('\nLoad flow data...')
    if mode == 'train':
        train_od, train_intensity = read_flows(os.path.join(file_path, 'train.txt'), entity2id)
        valid_od, valid_intensity = read_flows(os.path.join(file_path, 'valid.txt'), entity2id)

        print('\t\t# train flows: {}, # validation flows: {}'.format(len(train_od), len(valid_od)))

        train_batches = batch_generator(train_od, train_intensity, batch_size)
        return train_batches, valid_od, valid_intensity
    elif mode == 'test':
        test_od, test_intensity = read_flows(os.path.join(file_path, 'test.txt'), entity2id)

        print('\t# test flows: {}'.format(len(test_od)))
        return test_od, test_intensity
    else:
        raise Exception('Wrong mode.')


def read_flows(file_path, entity2id):
    od = []
    intensity = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            head, tail, m = line.strip().split('\t')
            od.append((entity2id[head], 0, entity2id[tail]))
            intensity.append(float(m))
    return torch.from_numpy(np.array(od)), torch.from_numpy(np.array(intensity).astype(np.float32))


def negative_sampling(pos_samples, pos_labels, entity_set, negative_num, exclusive=False):
    print('Sample negative flows...')
    candidate = []
    if exclusive:
        known_edges = set()
        for edge in pos_samples:
            known_edges.add((edge[0], edge[2]))

        for i in entity_set:
            for j in entity_set:
                if i != j and (i, j) not in known_edges:
                    candidate.append([i, 0, j])
    else:
        for i in entity_set:
            for j in entity_set:
                if i != j:
                    candidate.append([i, 0, j])

    candidate_num = len(candidate)
    negative_num = negative_num if negative_num < candidate_num else candidate_num
    idx = np.random.choice(candidate_num, size=negative_num, replace=False)
    neg_samples = np.array(candidate)[idx]

    samples = np.concatenate((pos_samples, neg_samples))
    labels = np.zeros(samples.shape[0], dtype=np.float32)
    labels[:pos_samples.shape[0]] = pos_labels[:]

    return torch.from_numpy(samples), torch.from_numpy(labels)


def batch_generator(triplets, labels, batch_size=0):
    if batch_size <= 0 or batch_size > triplets.shape[0]:
        batch_size = triplets.shape[0]

    return DataLoader(TensorDataset(triplets, labels), batch_size=batch_size, shuffle=True, num_workers=4)


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes=2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm


def build_embedding_graph(num_nodes, num_rels, triplets):
    print('Build embedding graph...')
    src, rel, dst = triplets.transpose()
    src = torch.from_numpy(src)
    rel = torch.from_numpy(rel)
    dst = torch.from_numpy(dst)

    # Trick: Create a bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    edge_type = torch.cat((rel, rel+num_rels))
    edge_index = torch.stack((src, dst))

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data
