# -*- coding: utf-8 -*-
"""
author: Xin Yao
create date: 2021-12-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from utils import uniform

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score, mean_absolute_error
import numpy as np
import pandas as pd
import scipy


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, embedding_size, dropout, bias=0, seed=None):
        if seed:
            setup_seed(seed)
        super(RGCN, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_size)
        self.relation_embedding = nn.Parameter(torch.Tensor(1, embedding_size, embedding_size))
        self.bias = nn.Parameter(torch.Tensor(1))

        nn.init.xavier_uniform_(self.relation_embedding)  # gain=nn.init.calculate_gain('relu')
        nn.init.constant_(self.bias, bias)

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv = RGCNConv(
            embedding_size, embedding_size, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = self.conv(x, edge_index, edge_type, edge_norm)
        
        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]].unsqueeze(1)
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]].unsqueeze(2)
        score = torch.matmul(s, r)
        score = torch.matmul(score, o).squeeze()
        score = F.relu(score + self.bias)

        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.mse_loss(score, target)

    def score_mse(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)
        print(target.detach().numpy(), score.detach().numpy())
        return mean_squared_error(target.detach().numpy(), score.detach().numpy())

    def score_mape(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return np.mean(np.abs((score.detach().numpy() - target.detach().numpy()) / target.detach().numpy()))

    def score_cpc(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)
        y_true = list(target.detach().numpy())
        y_pred = list(score.detach().numpy())
        flow_min = 0
        flow_sum = 0
        for i in range(len(y_true)):
            flow_min = flow_min + min(y_true[i], y_pred[i])
            flow_sum = flow_sum + (y_true[i] + y_pred[i])
        return 2 * flow_min / flow_sum

    def score_mae(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return mean_absolute_error(target.detach().numpy(), score.detach().numpy())

    def score_evs(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return explained_variance_score(target.detach().numpy(), score.detach().numpy())

    def score_r2(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return r2_score(target.detach().numpy(), score.detach().numpy())

    def score_scc(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return scipy.stats.spearmanr(target.detach().numpy(), score.detach().numpy())[0]

    def score_scc_pvalue(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return scipy.stats.spearmanr(target.detach().numpy(), score.detach().numpy())[1]

    def export_results(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)
        df = pd.DataFrame()
        df['pred'] = score.detach().numpy()
        df['obs'] = target.detach().numpy()
        return df

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # lookup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                aggr_out += self.root
            else:
                aggr_out += torch.matmul(x, self.root)

        if self.bias is not None:
            aggr_out += self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
