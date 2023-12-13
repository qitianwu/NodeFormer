import json
import networkx as nx
import os
import numpy as np
from networkx.readwrite import json_graph
import torch

class NCDataset(object):
    def __init__(self, name):

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_trojan_data(prefix):
    G_data = json.load(open(prefix + "-G.json"))

    G = json_graph.node_link_graph(G_data)
    # edge_index
    # extract source and target nodes
    source = []
    target = []
    for edge in G.edges():
        source.append(edge[0])
        target.append(edge[1])
    source = torch.tensor(source)
    target = torch.tensor(target)
    edge_index = torch.stack((source, target), dim=0)

    # node_feat
    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None    
    node_feat = torch.tensor(feats,dtype=torch.int32)  # Convert feats to torch.Tensor
    print(node_feat)
    # num_nodes
    num_nodes = G.number_of_nodes()
    # print(num_nodes)
    # val_mask
    val_nodes = []

    for node in G.nodes(data=True):
        if node[1].get('val', False):
            val_nodes.append(node[0])
    val_mask = torch.tensor(val_nodes)
    print(val_mask)
    # test_mask
    test_nodes = []

    for node in G.nodes(data=True):
        if node[1].get('test', False):
            test_nodes.append(node[0])
    test_mask = torch.tensor(test_nodes)

    # train_mask
    filtered_nodes = []

    for node in G.nodes(data=True):
        if not node[1].get('val', True) and not node[1].get('test', True):
            filtered_nodes.append(node[0])
    train_mask = torch.tensor(filtered_nodes)

    # label
    class_map = json.load(open(prefix + "-class_map.json"))
    label = torch.tensor(list(class_map.values()))
    
    dataset = NCDataset(prefix)
    dataset.train_idx = train_mask
    dataset.valid_idx = val_mask
    dataset.test_idx = test_mask

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset

load_trojan_data("s38417t300")