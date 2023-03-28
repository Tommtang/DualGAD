import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv)

from torch_sparse import SparseTensor


def ce_loss(pos_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    # neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss

def edgeidx2sparse(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def creat_gnn_layer(name, first_channels, second_channels):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(
            nn.Sequential(nn.Linear(first_channels, second_channels)), train_eps=True
        )
    elif name == "gat":
        layer = GATConv(first_channels, second_channels, heads=1)
    elif name == "gat2":
        layer = GATv2Conv(first_channels, second_channels, heads=1)
    else:
        raise ValueError(name)
    return layer


def create_input_layer(num_nodes, num_node_feats,
                       use_node_feats=True, node_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if node_emb:
            emb = torch.nn.Embedding(num_nodes, node_emb)
            input_dim += node_emb
    else:
        emb = torch.nn.Embedding(num_nodes, node_emb)
        input_dim = node_emb
    return input_dim, emb


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")


class EdgeEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        bn=False,
        layer="gcn",
        activation="elu",
        use_node_feats=True,
        num_nodes=None,
        node_emb=None,
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.use_node_feats = use_node_feats
        self.node_emb = node_emb

        if node_emb is not None and num_nodes is None:
            raise RuntimeError("Please provide the argument `num_nodes`.")

        in_channels, self.emb = create_input_layer(
            num_nodes, in_channels, use_node_feats=use_node_feats, node_emb=node_emb
        )

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels))

            self.bns.append(bn(second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

        if self.emb is not None:
            nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, x):
        if self.use_node_feats:
            input_feat = x
            if self.node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def forward(self, x, edge_index):
        # x = self.create_input_feat(x)
        # edge_index = edgeidx2sparse(edge_index, x.size(0))

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            # x = self.bns[i](x)
            x = self.activation(x)
            # x = self.dropout(x)
        # x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        # x = self.bns[-1](x)
        # x = self.activation(x)
        return x
class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=True):
        # edge = torch.tensor(edge).type(torch.long)
        edge = edge.clone().detach().type(torch.long)
        # print('z', z.shape,edge)

        x = z[edge[0]] * z[edge[1]]
        # print('x1',x.shape,edge.shape)

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = mlp(x)
            x = self.activation(x)
            # x = self.dropout(x)



        x = self.mlps[-1](x)
        if sigmoid:
            return x.sigmoid()
        else:
            return x
