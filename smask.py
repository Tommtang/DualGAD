from utils import *
from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling
import torch
def choose(edge_index,n, mask_edges):
    # print('edge_index', edge_index, edge_index.shape)
    edge = negative_sampling(
        edge_index, num_nodes=n,
        num_neg_samples=mask_edges.view(2, -1).size(1)).view_as(mask_edges)
    # print('edge,edge.shape',mask_edges.shape,edge.shape)
    return edge

def mask_edge(mask, adj,  rawf, idx):
    adj = adj[idx, :][:, idx]
    index_dense = dense_to_sparse(adj)
    # index_dense = torch.tensor(index_dense).type(torch.long)
    if index_dense.shape[1] != 0:
        remaining_edges, mask_edges = mask(index_dense, idx)
        neg_edge_index = choose(index_dense, len(idx), mask_edges).type(torch.long)
        # print('mask_edges,neg_edge_index',mask_edges.shape,neg_edge_index.shape)
        tag = 1
        F_e_s = rawf[:, idx, :]
        F_e_s = torch.squeeze(F_e_s, 0)
        return remaining_edges, mask_edges, neg_edge_index,F_e_s, tag
    return 0, 0, 0, 0,0
