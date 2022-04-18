# -*- coding: utf-8 -*-
"""
author: Xin Yao
create date: 2021-12-02
"""

import argparse
import torch

from utils import load_graph_data, load_flow_data, read_setting
from models import RGCN
import os
import shutil


def evaluate(args, file_mark):
    setting = read_setting(args.setting)

    entity2id, relation2id, embedding_graph = load_graph_data(args.dataset, verbose=0)
    test_samples, test_labels = load_flow_data(args.dataset, entity2id, mode='test')

    model = RGCN(len(entity2id), len(relation2id), num_bases=int(setting['n_bases']),
                 embedding_size=int(setting['embedding_size']), dropout=float(setting['dropout']))

    checkpoint = torch.load('saved_model/'+args.model)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    entity_embedding = model(embedding_graph.entity, embedding_graph.edge_index, embedding_graph.edge_type,
                             embedding_graph.edge_norm)

    rmse = model.score_rmse(entity_embedding, test_samples, test_labels)
    mape = model.score_mape(entity_embedding, test_samples, test_labels)
    cpc = model.score_cpc(entity_embedding, test_samples, test_labels)
    r2 = model.score_r2(entity_embedding, test_samples, test_labels)

    model.export_results(entity_embedding, test_samples, test_labels).to_csv('result/'+file_mark+'_eval.csv',
                                                                             header=True, encoding='gbk')

    return [rmse, mape, cpc, r2]


if __name__ == '__main__':
    with open('metrics.csv', 'w') as f:
        f.write('model,rmse,mape,cpc,r2\r\n')
        for file in os.listdir('saved_model/'):
            if file[-3:] == 'pth':
                print('\n------------------------------------')
                print('Model:', file)
                file_mark = file[0:-4]
                # Attention: Replace the graph file that is used for training
                # shutil.copy2('data/graph_knn/graph_30.txt', 'data/mobility/graph.txt')

                parser = argparse.ArgumentParser(description='RGCN')
                parser.add_argument("--dataset", type=str, default="./data/mobility")
                parser.add_argument("--setting", type=str, default="./setting.txt")
                parser.add_argument("--model", type=str, default=file)

                args = parser.parse_args()
                # print('Data path:')
                # print('\t' + args.dataset)
                # print('Setting file:')
                # print('\t' + args.setting)

                r = evaluate(args, file_mark)
                print('Metrics:')
                print('\t[rmse, mape, cpc, r2]=', r)
                f.write(','.join([file_mark] + list(map(str, r)))+'\r\n')
