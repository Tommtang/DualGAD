import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
import random
import dgl
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected, k_hop_subgraph
import numpy as np
import scipy
from torch_sparse import SparseTensor

def subgraph(edge_index, idx, k_hop=2):
    subset, edge_index, inv, edge_mask=k_hop_subgraph(
        node_idx=idx, num_hops=k_hop,
        edge_index=edge_index,
        # relabel_nodes=True,
        # num_nodes=graph.num_nodes
    )
    print('subgraph',edge_index.shape,edge_mask.shape)

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def edgeidx2sparse(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # print(sparse_mx)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    r= torch.sparse.FloatTensor(indices, values, shape)
    return indices,r

def dense_to_sparse(adj):
    # adj_dense = adj.todense()
    adj_sp = scipy.sparse.csr_matrix(adj)
    # print('adj_sp',adj_sp)
    index,r = sparse_mx_to_torch_sparse_tensor(adj_sp)
    return index

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    # print('adj', type(adj))
    ins=adj
    # print('ins',ins)
    index = sparse_mx_to_torch_sparse_tensor(adj)
    labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels,num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    num_test = num_node - num_train - num_val
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, num_train, num_val, num_test, ano_labels, str_ano_labels, attr_ano_labels

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

# import dgl.contrib.sampling
# import dgl.sampling.randomwalks as ss
def generate_rwr_subgraph(dgl_graph, subgraph_size):
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1

    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size * 3)
    subv = []

    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv

def mask_path(edge_index,idx,
              walks_per_node: int = 2,
              walk_length: int = 8, r: float = 0.2,
              p: float = 1, q: float = 1, num_nodes=None,
              by='degree',
              replacement=True):
    # print('edge_index.shape',edge_index.shape)
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    # print('num_nodes',num_nodes)

    assert by in {'degree', 'uniform'}

    row = edge_index[0]
    col = edge_index[1]
    deg = torch.zeros(num_nodes, device=row.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=row.device))

    if isinstance(r, float):
        assert 0 < r <= 1
        num_starts = int(r * num_nodes)
        if by == 'degree':
            prob = deg.float() / deg.sum()
            start = prob.multinomial(num_samples=num_starts, replacement=replacement)
        else:
            start = torch.randperm(num_nodes, device=edge_index.device)[:num_starts]
    elif torch.is_tensor(r):
        start = r.to(edge_index)
        n = start.size(0)
        start = start[torch.randperm(n)[:n // 3]]
    else:
        raise ValueError(r)

    deg = row.new_zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones_like(row))
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, p, q)
    mask = row.new_ones(row.size(0), dtype=torch.bool)
    mask[e_id.view(-1)] = False
    return edge_index[:, mask], edge_index[:, e_id]

class MaskPath(nn.Module):
    def __init__(self, walks_per_node: int = 2,
                 walk_length: int = 3, r: float = 0.8,
                 p: float = 1, q: float = 1, num_nodes=None,
                 by='degree', undirected=True, replacement=True):
        super(MaskPath, self).__init__()
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.r = r
        self.num_nodes = num_nodes
        self.by = by
        self.undirected = undirected
        self.replacement = replacement

    def forward(self, edge_index,idx, length=8):
        remaining_edges, masked_edges = mask_path(edge_index,idx,
                                                  walks_per_node=self.walks_per_node,
                                                  walk_length=self.walk_length,
                                                  p=self.p, q=self.q,
                                                  r=self.r, num_nodes=self.num_nodes, by=self.by, replacement=self.replacement)
        remaining_edges = remaining_edges.cuda()

        masked_edges=np.array(masked_edges).astype(int)
        masked_edges=torch.from_numpy(masked_edges)
        masked_edges = masked_edges.cuda()

        # print('remaining_edges masked_edges 1', masked_edges, masked_edges.shape)
        # print('remaining_edges masked_edges 11', remaining_edges, remaining_edges.shape)
        # if self.undirected:
        # remaining_edges = to_undirected(remaining_edges)

        # print('remaining_edges masked_edges 2',remaining_edges, remaining_edges.shape)
        # remaining_edges = scipy.sparse.coo_matrix(remaining_edges)
        # print('remaining_edges',remaining_edges, type(remaining_edges))
        # re_adj = edgeidx2sparse(remaining_edges, length).to_dense()
        # print('edge_index1', re_adj.shape)

        return remaining_edges, masked_edges

    def extra_repr(self):
        return f"walks_per_node={self.walks_per_node}, walk_length={self.walk_length}, \n"\
            f"r={self.r}, p={self.p}, q={self.q}, by={self.by}, undirected={self.undirected}, replacement={self.replacement}"


def mask_edge(edge_index, p=0.7):
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]


class MaskEdge(nn.Module):
    def __init__(self, p=0.7, undirected=True):
        super(MaskEdge, self).__init__()
        self.p = p
        self.undirected = undirected

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        # print('===========1', remaining_edges.shape, masked_edges.shape)
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        # print('===========',remaining_edges.shape,masked_edges.shape)
        return remaining_edges, masked_edges

    def extra_repr(self):
        return f"p={self.p}, undirected={self.undirected}"
