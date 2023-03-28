import torch
from torch import nn
from layer import encoder, decoder, GCN, MLP_Predictor, AvgReadout
from function import EdgeDecoder, EdgeEncoder
import torch.nn.functional as F
import copy
import numpy as np
import faiss
from torch.nn.functional import cosine_similarity
def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def kms(target, up_q, down_q):
    index = faiss(n_clusters=2, random_state=0).fit(target.detach().cpu().numpy())
    ss_label = index.labels_
    cluster_info = [list(np.where(ss_label == i)[0]) for i in range(2)]
    kl_loss=0
    for i in range(2):
        node_idx = cluster_info[i]
        up = up_q[node_idx]
        down = down_q[node_idx]
        h_1_block = torch.unsqueeze(target[node_idx], 0)
        c_block = torch.sum(h_1_block)
        c_block = c_block.expand_as(up)
        kl_loss = kl_loss + cosine_similarity(up, c_block.detach(), dim=-1).mean() + cosine_similarity(down,c_block.detach(),dim=-1).mean()
    kl_loss = 4 - kl_loss
    return  kl_loss



class DualGAD(nn.Module):
    def __init__(self, nfeat, nhid,nclass=1, dropout=0.5):
        super(DualGAD, self).__init__()
        self.enc_n = encoder(nfeat, nhid)
        self.dec_n = decoder(nhid, nfeat)

        self.enc_e = EdgeEncoder(nfeat, nhid, nhid)
        self.dec_e = EdgeDecoder(nhid, nfeat)

        self.mlp1 = MLP_Predictor(nhid,nhid)

        self.mlp2 = MLP_Predictor(nhid, nhid)

        self.mlp3 = MLP_Predictor(nhid, nhid)

        # target network
        self.target_encoder = copy.deepcopy(GCN(nfeat, nhid))

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.pool = AvgReadout()
        self.pdist = nn.PairwiseDistance(p=2)
        self.mse_loss = nn.MSELoss(reduction='mean')


    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.enc_n.parameters()) + list(self.dec_n.parameters()) + list(self.enc_e.parameters()) + list(self.dec_e.parameters()) + list(self.dec_e.parameters()) \
            + list(self.mlp1.parameters()) + list(self.mlp2.parameters()) + list(self.mlp3.parameters()) + list(self.target_encoder.parameters())


    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_p, param_t in zip(self.enc_n.parameters(), self.enc_e.parameters(),self.target_encoder.parameters()):
            param_t.data.mul_(mm).add_(param_q.data + param_p.data, alpha=1. - mm)

    def forward(self,F_n ,  A_n,of, oa, F_e, A_e, masked_edges,neg_mas,alpha):
        emb_f = self.enc_n(F_n , A_n)
        z_f= self.dec_n(emb_f,A_n)
        emb_s = self.enc_e(F_e, A_e)
        pos_out = self.dec_e(emb_s, masked_edges)
        neg_out = self.dec_e(emb_s, neg_mas)
        emb_f = torch.sum(emb_f, dim=1)
        online1_p = self.mlp1(emb_f)
        online2_p = self.mlp2(emb_s)
        with torch.no_grad():
            target = self.target_encoder(of,oa).detach()
            target= torch.squeeze(target,0)
        target = self.mlp3(target)
        kl_loss = kms(target, online1_p, online2_p)
        loss_edge = ce_loss(pos_out, neg_out)
        loss_node = self.mse_loss(z_f[:, -2, :], F_n[:, -1, :])
        return alpha*(loss_node+loss_edge) + (1-alpha)*kl_loss

    def inference(self,F_n ,  A_n,of, oa, F_e, A_e,  adj_t,alpha):
        emb_n = self.enc_n(F_n, A_n)
        z_n = self.dec_n(emb_n, A_n)

        out = self.enc_e(F_e, A_e)
        A_re = out @ out.T
        diff_structure = torch.pow(A_re - adj_t, 2)
        score2 = torch.sqrt(torch.sum(diff_structure, 1))

        score1 = self.pdist(z_n[:, -2, :], F_n[:, -1, :])
        return score1 + score2

