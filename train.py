# -*- coding: utf-8 -*-
"""
author: Xin Yao
create date: 2021-12-02
"""

import argparse
import numpy as np
import torch
import time
import datetime
import pandas as pd

from utils import load_graph_data, load_flow_data, read_setting
from models import RGCN


def main(args):
    setting = read_setting(args.setting)

    use_cuda = int(setting['gpu']) >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(setting['gpu'])

    entity2id, relation2id, embedding_graph = load_graph_data(args.dataset)
    train_batches, valid_samples, valid_labels \
        = load_flow_data(args.dataset, entity2id, mode='train', batch_size=int(setting['batch_size']))

    print('\nCreate model...')
    seed = None
    if 'seed' in setting:
        seed = int(setting['seed'])
    model = RGCN(len(entity2id), len(relation2id), num_bases=int(setting['n_bases']),
                 embedding_size=int(setting['embedding_size']), dropout=float(setting['dropout']), seed=seed)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(setting['learning_rate']))

    if use_cuda:
        model.cuda()

    print('\nStart training...')
    best_loss = np.inf
    saved_model_name = args.dataset.split('/')[-1] + '_es' + setting['embedding_size'] + '_' + \
                       datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')
    df_losses = pd.DataFrame()
    ls_train_loss = []
    ls_valid_loss = []
    for epoch in range(1, (int(setting['n_epochs']) + 1)):
        print('Start Epoch ---%d--- at time: %s' % (epoch, str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))))
        model.train()
        train_losses = []

        for samples, labels in train_batches:
            if use_cuda:
                device = torch.device('cuda')
                samples.to(device)
                labels.to(device)

            entity_embedding = model(embedding_graph.entity, embedding_graph.edge_index, embedding_graph.edge_type,
                                     embedding_graph.edge_norm)
            loss = model.score_loss(entity_embedding, samples, labels)
            loss += float(setting['regularization']) * model.reg_loss(entity_embedding)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(setting['grad_norm']))
            optimizer.step()
            train_losses.append(loss)

        if epoch % int(setting['evaluate_every']) == 0:
            print("\tepoch {}: train loss = {}".format(epoch, sum(train_losses)/len(train_losses)))

            if use_cuda:
                model.cpu()

            model.eval()

            entity_embedding = model(embedding_graph.entity, embedding_graph.edge_index, embedding_graph.edge_type,
                                     embedding_graph.edge_norm)
            valid_loss = model.score_loss(entity_embedding, valid_samples, valid_labels)

            if valid_loss < best_loss:
                print('\t\tloss decreased {:.4f} --> {:.4f}. saving model...'.format(best_loss, valid_loss))
                best_loss = valid_loss
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, 'output/'+saved_model_name+'.pth')

            if use_cuda:
                model.cuda()
            ls_train_loss.append((sum(train_losses)/len(train_losses)).detach().numpy())
            ls_valid_loss.append(best_loss.detach().numpy())

    file_name = saved_model_name + '_losses'
    df_losses['train_loss'] = ls_train_loss
    df_losses['valid_loss'] = ls_valid_loss
    df_losses.to_csv('output/'+file_name+'.csv', header=True, index=False, encoding='gbk')


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dataset", type=str, default="./data/test")
    parser.add_argument("--setting", type=str, default="./setting.txt")

    args = parser.parse_args()
    print('Data path:')
    print('\t' + args.dataset)
    print('Setting file:')
    print('\t' + args.setting)

    main(args)

    running_time = (time.time() - start_time) / 60
    print('\nCreated and saved models successfully!')
    print('Total time cost: %.2f mins' % running_time)

