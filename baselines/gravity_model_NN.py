# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import datetime
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from func import *


def load_data(path, batch_size):
    train_f = read_flows([path + 'train.txt'])  # Flow list: [origin, destination, intensity]
    validation_f = read_flows([path + 'valid.txt'])
    test_f = read_flows([path + 'test.txt'])
    features = read_features(path + 'features_pop.txt', normalized=False)  # Feature dict: [x/lng, y/lat, push, pull]

    train_X = []
    train_y = []
    valid_X = []
    valid_y = []
    test_X = []
    test_y = []

    for k in train_f:
        train_y.append(k[2])
        train_X.append([np.log(features[k[0]][2]), np.log(features[k[1]][2]),
                        np.log(haversine_distance(features[k[0]][0], features[k[0]][1], features[k[1]][0], features[k[1]][1]))])
        # train_X.append([features[k[0]][2], features[k[1]][2],
        #                 haversine_distance(features[k[0]][0], features[k[0]][1], features[k[1]][0], features[k[1]][1])/1000])

    for k in test_f:
        test_y.append(k[2])
        test_X.append([np.log(features[k[0]][2]), np.log(features[k[1]][2]),
                       np.log(haversine_distance(features[k[0]][0], features[k[0]][1], features[k[1]][0], features[k[1]][1]))])
        # test_X.append([features[k[0]][2], features[k[1]][2],
        #                haversine_distance(features[k[0]][0], features[k[0]][1], features[k[1]][0], features[k[1]][1])/1000])

    for k in validation_f:
        valid_y.append(k[2])
        valid_X.append([np.log(features[k[0]][2]), np.log(features[k[1]][2]),
                        np.log(haversine_distance(features[k[0]][0], features[k[0]][1], features[k[1]][0], features[k[1]][1]))])
        # valid_X.append([features[k[0]][2], features[k[1]][2],
        #                 haversine_distance(features[k[0]][0], features[k[0]][1], features[k[1]][0], features[k[1]][1])/1000])

    train_batches = DataLoader(TensorDataset(torch.from_numpy(np.array(train_X).reshape((-1, 3)).astype(np.float32)),
                                             torch.from_numpy(np.array(train_y).reshape((-1, 1)).astype(np.float32))),
                               batch_size=batch_size, shuffle=True, num_workers=0)

    return train_batches, \
           torch.tensor(np.array(test_X).reshape((-1, 3)).astype(np.float32)), \
           np.array(test_y), \
           torch.tensor(np.array(valid_X).reshape((-1, 3)).astype(np.float32)), \
           torch.tensor(np.array(valid_y).reshape((-1, 1)).astype(np.float32))


def train(model, train_batches, valid_X, valid_y, settings, saved_model_name):
    use_cuda = int(settings['gpu']) >= 0 and torch.cuda.is_available()
    device = None
    if use_cuda:
        torch.cuda.set_device(settings['gpu'])
        device = torch.device('cuda')
        model.cuda()

    print('\nStart training...')
    best_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings['learning_rate']))
    criterion = nn.MSELoss(reduction='mean').to(device)
    for epoch in range(1, (int(settings['n_epochs']) + 1)):
        print('epoch:', epoch)
        model.train()
        train_losses = []

        for features, labels in train_batches:
            if use_cuda:
                features.to(device)
                labels.to(device)

            pred = model(features)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings['grad_norm']))
            optimizer.step()
            train_losses.append(loss.detach().numpy())

        if epoch % int(settings['evaluate_every']) == 0:
            print('\t' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                  ' epoch {}: train loss = {}'.format(epoch, np.mean(train_losses)))

            if use_cuda:
                valid_X.to(device)
                valid_y.to(device)

            model.eval()

            pred = model(valid_X)
            valid_loss = criterion(pred, valid_y)
            if valid_loss < best_loss:
                print('\t\tloss decreased {:.4f} --> {:.4f}. saving model...'.format(best_loss, valid_loss))
                best_loss = valid_loss
                torch.save(model.state_dict(), saved_model_name)
            if use_cuda:
                model.cuda()


class GMNN(nn.Module):
    def __init__(self, hidden_size):
        super(GMNN, self).__init__()
        self.fc1 = nn.Linear(3, hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


def train_func(data_path, settings):
    with open('gmnn_result.csv', 'w') as f:
        f.write('model,rmse,mape,cpc,smc,r2\r\n')
        for i in range(25):
            saved_model_name = 'GMNN_'+str(i)+'.pt'
            train_batches, test_X, test_y, valid_X, valid_y = load_data(data_path, settings['batch_size'])
            print(test_X.shape, valid_X.shape, valid_y.shape)

            model = GMNN(settings['hidden_size'])

            train(model, train_batches, valid_X, valid_y, settings, 'GMNN_models_pop/' + saved_model_name)

            # Evaluate model
            model.eval()
            with torch.no_grad():
                pred = torch.squeeze(model(test_X)).detach().numpy()
                m = evaluate(pred, test_y)
                result = np.concatenate((test_y.reshape((-1, 1)), pred.reshape((-1, 1))), axis=1)
                np.savetxt('pred_'+saved_model_name+'.txt', result, delimiter=',')

                f.write(str(i)+','+','.join(list(map(str, m)))+'\r\n')


def evaluate_func(model_path, data_path, settings):
    _, test_X, test_y, _, _ = load_data(data_path, settings['batch_size'])

    model = GMNN(settings['hidden_size'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        pred = torch.squeeze(model(test_X)).detach().numpy()
        m = evaluate(pred, test_y)
        # result = np.concatenate((np.array(test_y.numpy()).reshape((-1, 1)), np.array(pred).reshape((-1, 1))), axis=1)
        # np.savetxt('pred_'+model[:-4]+'.txt', result, delimiter=',')


if __name__ == '__main__':
    data_path = '../data/'

    # Use the same parameters as GCN if possible
    settings = {
        'batch_size': 256,
        'learning_rate': 0.005,
        'hidden_size': 10,
        'gpu': -1,
        'n_epochs': 250,
        'evaluate_every': 1,
        'grad_norm': 1.0
    }
    train_func(data_path, settings)
    # evaluate_func('GMNN_models_pop/GMNN_18.pth', data_path, settings)
