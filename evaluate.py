# -*- coding: utf-8 -*-
"""
author: Xin Yao
create date: 2021-12-02
"""

import argparse
import torch

from utils import load_graph_data, load_flow_data, read_setting
from models import RGCN
import numpy as np
import os


def main(args, file_mark):
    setting = read_setting(args.setting)

    entity2id, relation2id, embedding_graph = load_graph_data(args.dataset)
    test_samples, test_labels = load_flow_data(args.dataset, entity2id, mode='test')

    model = RGCN(len(entity2id), len(relation2id), num_bases=int(setting['n_bases']),
                 embedding_size=int(setting['embedding_size']), dropout=float(setting['dropout']))

    checkpoint = torch.load('output/'+args.model)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    entity_embedding = model(embedding_graph.entity, embedding_graph.edge_index, embedding_graph.edge_type,
                             embedding_graph.edge_norm)

    mse = model.score_mse(entity_embedding, test_samples, test_labels)
    # print('\n MSE of test data: {}'.format(mse))

    rmse = np.sqrt(mse)
    # print('\n RMSE of test data: {}'.format(rmse))

    mape = model.score_mape(entity_embedding, test_samples, test_labels)
    # print('\n MAPE of test data: {}'.format(mape))

    cpc = model.score_cpc(entity_embedding, test_samples, test_labels)
    # print('\n CPC of test data: {}'.format(cpc))

    mae = model.score_mae(entity_embedding, test_samples, test_labels)
    # print('\n MAE of test data: {}'.format(mae))

    evs = model.score_evs(entity_embedding, test_samples, test_labels)
    # print('\n EVS of test data: {}'.format(evs))

    r2 = model.score_r2(entity_embedding, test_samples, test_labels)
    # print('\n R2 of test data: {}'.format(r2))

    scc = model.score_scc(entity_embedding, test_samples, test_labels)
    # print('\n SCC of test data: {}'.format(scc))

    # file_name = args.dataset.split('/')[-1] + '_' + 'metric' + '_' + setting['negative_num'] + '_' + \
    #             setting['n_bases'] + '_' + setting['embedding_size']

    # print flow dataframe
    model.export_results(entity_embedding, test_samples, test_labels).to_csv('output/'+file_mark+'_pred.csv',
                                                                             header=True, encoding='gbk')
    # print attribute matrix
    # with open('./result/'+file_mark+'_attr.txt', 'wb') as f:
    #     for line in np.matrix(checkpoint['state_dict']['conv.att'].detach().numpy()):
    #         np.savetxt(f, line, fmt='%.4f')

    return [mse, rmse, mape, cpc, mae, evs, r2, scc]


if __name__ == '__main__':
    with open('output/metrics_all.csv', 'w') as f:
        f.write('model,mse,rmse,mape,cpc,mae,evs,r2,scc\r\n')
        for file in os.listdir('output/'):
            if file[-3:] == 'pth':
                print('\n' + file)
                file_mark = file[0:-4]
                parser = argparse.ArgumentParser(description='RGCN')
                parser.add_argument("--dataset", type=str, default="./data/test")
                parser.add_argument("--setting", type=str, default="./setting.txt")
                parser.add_argument("--model", type=str, default=file)

                args = parser.parse_args()
                print('Data path:')
                print('\t' + args.dataset)
                print('Setting file:')
                print('\t' + args.setting)

                r = main(args, file_mark)
                f.write(','.join([file_mark] + list(map(str, r)))+'\r\n')   # Change \r\n if necessary
