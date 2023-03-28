import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
        self.reset_parameters()

    def forward(self, x, adj):
        # x = self.gc1(x, adj)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x
    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        # self.gc3.reset_parameters()

class Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(Decoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nhid)
        # self.dropout = dropout
        self.reset_parameters()
    def forward(self, x, adj):
        x= self.gc1(x, adj)
        return x
    def reset_parameters(self):
        self.gc1.reset_parameters()


class GCN(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """

    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.ReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return out



class encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(encoder, self).__init__()
        self.gc1 = GCN(nfeat, nhid)
        self.gc2 = GCN(nhid, nhid)
        self.dropout = dropout
        self.reset_parameters()


    def forward(self, x, adj):
        x = self.gc1(x,adj)
        x =F.relu(x)
        return x
    def reset_parameters(self):
        self.gc1.reset_parameters()

class decoder(nn.Module):
    def __init__(self, nhid,nfeat, dropout=0.5):
        super(decoder, self).__init__()
        # self.gc1 = GCN(nhid, nhid)
        self.gc2= GCN(nhid, nfeat)
        self.dropout = dropout
        self.reset_parameters()

    def forward(self, x, adj):
        x = self.gc2(x, adj)
        return x
    def reset_parameters(self):
        self.gc2.reset_parameters()
        # self.gc2.reset_parameters()

class decoder_down(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(decoder_down, self).__init__()

        self.gc1 = GCN(nfeat, nhid)

    def forward(self, x, adj):
        x = self.gc1(x,adj)
        return x
    def reset_parameters(self):
        self.gc1.weights_init(nn.Linear)

class decoder_up(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(decoder_up, self).__init__()

        self.gc1 = GCN(nfeat, nhid)

    def forward(self, x, adj):
        x = self.gc1(x,adj)
        return x
    def reset_parameters(self):
        self.gc1.weights_init(nn.Linear)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=128):
        super().__init__()

        self.net = nn.Sequential(
            # nn.Linear(input_size, hidden_size, bias=True),
            # nn.PReLU(1),
            # nn.BatchNorm1d(hidden_size),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_size, output_size, bias=True)
            nn.Linear(input_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()