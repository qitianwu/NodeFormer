import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, k_hop_subgraph
from torch_scatter import scatter

from nodeformer import *
from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits, adj_mul, to_sparse_tensor
from eval import evaluate_cpu, eval_acc, eval_rocauc, eval_f1
from parse import parse_method, parser_add_main_args
import time

import warnings

warnings.filterwarnings('ignore')


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


fix_seed(42)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

device = torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

# get the splits for all runs
split_idx = dataset.load_fixed_splits()

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']

print(f"num nodes {n} | num edges {edge_index.size(1)} | num classes {c} | num node feats {d}")

### Load method ###
model = NodeFormer(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                    num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                    use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
                    nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans).to(device)

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

adjs = []
adj, _ = remove_self_loops(edge_index)
adj, _ = add_self_loops(adj, num_nodes=n)
adjs.append(adj)
for i in range(args.rb_order - 1):
    adj = adj_mul(adj, adj, n)
    adjs.append(adj)

### Evaluation ###

print("Load model checkpoint...")
checkpoint_dir = f'../model/{args.dataset}-nodeformer.pkl'
checkpoint = torch.load(checkpoint_dir)
model.load_state_dict(checkpoint)

print(f"Evaluate the model on {args.dataset}...")
model.eval()
with torch.no_grad():
    out, _ = model(x, adjs)
    test_acc = eval_func(dataset.label[split_idx['test']], out[split_idx['test']])
if args.metric == 'rocauc':
    print(f'Test ROCAUC: {test_acc*100:.2f}%')
else:
    print(f'Test Accuracy: {test_acc*100:.2f}%')

