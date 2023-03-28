import os
from model import DualGAD
import argparse
from tqdm import tqdm
from utils import *
from utils import generate_rwr_subgraph, MaskPath
from schedule import  CosineDecayScheduler
from sklearn.metrics import roc_auc_score
from fmask import mask_attribute
from smask import mask_edge
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='DualGAD: Dual-bootstrapped Self-Supervised Learning for Graph Anomaly Detection')
parser.add_argument('--dataset', type=str, default='FinV')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--train_epoch', type=int, default=20)
parser.add_argument('--test_rounds', type=int, default=10)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--walks_per_node', type=int, default=3)
parser.add_argument('--wl', type=int, default=2)
parser.add_argument('--dr', type=int, default=0.5)
parser.add_argument('--p', type=int, default=1)
parser.add_argument('--q', type=int, default=1)
parser.add_argument('--expid', type=int)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--mm', type=float, default=0.99)
parser.add_argument('--lr_warmup_steps', type=int, default=1000)
parser.add_argument('--patience', type=int, default=20)

args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['FinV','YelpChi']:
        args.lr = 5e-3
    elif args.dataset == 'Elliptic':
        args.lr = 5e-4
    elif args.dataset == 'TeleCom':
        args.lr = 3e-3

batch_size = args.batch_size
subgraph_size = args.subgraph_size


# Load and preprocess data
adj, features, labels, num_train, num_val, num_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
raw_features = features.todense()
label_true = torch.Tensor(ano_label)
all_idx = list(range(features.shape[0]))
random.shuffle(all_idx)
idx_train = all_idx[ : num_train]
idx_val = all_idx[num_train : num_train + num_val]
idx_test = all_idx[num_train + num_val : ]

split_data = [num_train, num_val, num_test]
split_idx = [idx_train, idx_val, idx_test]

features, _ = preprocess_features(features)
dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

adj_dense = adj.toarray()


adj = (adj + sp.eye(adj.shape[0]))
adj = normalize_adj(adj).toarray()

features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
raw_features = torch.FloatTensor(raw_features[np.newaxis]).cuda()

# scheduler
lr_scheduler = CosineDecayScheduler(0.01, 50, 100)
mm_scheduler = CosineDecayScheduler(0.1 , 50, 100)

if torch.cuda.is_available():
    print('Using CUDA')
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()

cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size + 1

added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
added_adj_zero_col[:,-1,:] = 1.
added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))
if torch.cuda.is_available():
    added_adj_zero_row = added_adj_zero_row.cuda()
    added_adj_zero_col = added_adj_zero_col.cuda()
    added_feat_zero_row = added_feat_zero_row.cuda()


dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize model and optimiser
model = DualGAD(ft_size, args.embedding_dim).cuda()
optimiser = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

mask = MaskPath(walks_per_node=args.walks_per_node, walk_length=args.wl, r=args.dr, p=args.p, q=args.q)

best_r=0
for epoch in range(args.train_epoch):
    model.train()

    all_idx = list(range(nb_nodes))
    random.shuffle(all_idx)
    total_loss = 0.
    subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

    for batch_idx in range(batch_num):

        optimiser.zero_grad()

        is_final_batch = (batch_idx == (batch_num - 1))

        if not is_final_batch:
            idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        else:
            idx = all_idx[batch_idx * batch_size:]

        cur_batch_size = len(idx)

        F_n, A_n = mask_attribute(adj, features,raw_features,idx,subgraphs,args)
        A_e, mas, neg_mas, F_e, tag = mask_edge(mask, adj_dense, raw_features, idx)
        if tag == 0:
            continue
        oa = adj[:, idx, :][:, :, idx]
        of = features[:, idx, :]
        loss = model(F_n , A_n , of, oa, F_e, A_e,mas,neg_mas,args.alpha)
        loss.backward()
        optimiser.step()
        loss = loss.detach().cpu().numpy()
        if not is_final_batch:
            total_loss += loss

    mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
    if mean_loss < best:
        best = mean_loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_model.pkl')
    else:
        cnt_wait += 1
    if cnt_wait == args.patience:
        print('Early stopping!', flush=True)
        break
    print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)

model.load_state_dict(torch.load('best_model.pkl'))
print('Testing AUC!', flush=True)

multi_round_ano_score = np.zeros((args.test_rounds, nb_nodes))
for round in range(args.test_rounds):
    all_idx = list(range(nb_nodes))
    random.shuffle(all_idx)
    total_loss = 0.
    subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
    for batch_idx in range(batch_num):
        optimiser.zero_grad()
        is_final_batch = (batch_idx == (batch_num - 1))

        if not is_final_batch:
            idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        else:
            idx = all_idx[batch_idx * batch_size:]

        cur_batch_size = len(idx)

        F_n, A_n = mask_attribute(adj, features, raw_features, idx, subgraphs, args)

        A_e, mas, neg_mas, F_e, tag = mask_edge(mask, adj_dense, raw_features, idx)
        if tag == 0:
            continue
        adj_t = torch.tensor(adj_dense[idx, :][:, idx]).cuda()
        oa = adj[:, idx, :][:, :, idx]
        of = features[:, idx, :]
        with torch.no_grad():
            logits= model.inference(F_n, A_n, of, oa, F_e, A_e,  adj_t,args.alpha)
        ano_score = logits.cpu().numpy()
        multi_round_ano_score[round, idx] = ano_score
    ano_score = multi_round_ano_score[round,:]
    auc = roc_auc_score(ano_label, ano_score)
    print('Testing Epoch:{} Score:{:.8f}'.format(round, auc), flush=True)
ano_score_final = np.mean(multi_round_ano_score, axis=0)
auc = roc_auc_score(ano_label, ano_score_final)
print('Finally  Testing AUC:{:.4f}'.format(auc), flush=True)






















#         ### test
#         best_val_f1, test_f1= test(epoch_embedding,epoch)
#         print('Epoch:{} Loss:{:.8f} best_val_f1:{:.8f} test_f1:{:.8f}'.format(epoch, mean_loss, best_val_f1, test_f1),flush=True)
#
#         if best_r<test_f1:
#             best_r=test_f1
#             best_e=epoch
#
#     print('Loading {}th epoch, best results {} '.format(best_e, best_r))
#         # pbar.set_postfix(loss=mean_loss)
#         # pbar.update(1)
#
#
# # Test model
# print('Loading {}th epoch'.format(best_t))
# model.load_state_dict(torch.load('best_model.pkl'))



