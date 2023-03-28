import torch

def mask_attribute(adj, features,raw_features,idx,subgraphs,args):
    subgraph_size = args.subgraph_size
    ft_size = features.shape[-1]
    cur_batch_size = len(idx)
    ba = []
    bf = []
    F_e = []
    raw_bf =[]
    added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
    added_adj_zero_col[:, -1, :] = 1.
    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

    if torch.cuda.is_available():
        added_adj_zero_row = added_adj_zero_row.cuda()
        added_adj_zero_col = added_adj_zero_col.cuda()
        added_feat_zero_row = added_feat_zero_row.cuda()
    A_e = []
    mas = []
    for i in idx:
        # sub_adj  =  adj_dense[subgraphs[i], :][:, subgraphs[i]]
        cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
        cur_feat = features[:, subgraphs[i], :]
        raw_cur_feat = raw_features[:, subgraphs[i], :]
        ba.append(cur_adj)
        bf.append(cur_feat)
        F_e.append(cur_feat)
        raw_bf.append(raw_cur_feat)
        # A_e.append(a_e)
        # mas.append(a_m)
    # A_e = torch.cat(A_e).cuda()
    ba = torch.cat(ba)
    ba = torch.cat((ba, added_adj_zero_row), dim=1)
    ba = torch.cat((ba, added_adj_zero_col), dim=2)
    bf = torch.cat(bf)
    bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]) ,dim=1)
    raw_bf = torch.cat(raw_bf)
    raw_bf = torch.cat((raw_bf[:, :-1, :], added_feat_zero_row, raw_bf[:, -1:, :]), dim=1)
    A_n = ba
    F_n = raw_bf

    return F_n, A_n