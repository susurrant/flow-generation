# -*- coding: utf-8 -*-
"""
author: Xin Yao
create date: 2021-12-02
"""

import argparse
import torch

from utils import load_graph_data, load_flow_data, read_setting
from models import RGCN


def main(args):
    setting = read_setting(args.setting)

    entity2id, relation2id, embedding_graph = load_graph_data(args.dataset)
    test_samples, test_labels = load_flow_data(args.dataset, entity2id, mode='test')

    model = RGCN(len(entity2id), len(relation2id), num_bases=int(setting['n_bases']),
                 embedding_size=int(setting['embedding_size']), dropout=float(setting['dropout']))

    checkpoint = torch.load('./saved_model/'+args.model)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    entity_embedding = model(embedding_graph.entity, embedding_graph.edge_index, embedding_graph.edge_type,
                             embedding_graph.edge_norm)
    test_mse = model.score_loss(entity_embedding, test_samples, test_labels)

    print('\nMSE of test data: {}'.format(test_mse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dataset", type=str, default="./data/simple")
    parser.add_argument("--setting", type=str, default="./setting.txt")
    parser.add_argument("--model", type=str, default='simple_20211203-17-48-13.pth')

    args = parser.parse_args()
    print('Data path:')
    print('\t' + args.dataset)
    print('Setting file:')
    print('\t' + args.setting)

    main(args)
