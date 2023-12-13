from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T

from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url, class_rand_splits

from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import degree
import os

from google_drive_downloader import GoogleDriveDownloader as gdd

import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp

import json

from ogb.nodeproppred import NodePropPredDataset


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        split_type: 'random' for random splitting, 'class' for splitting with equal node num per class
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        label_num_per_class: num of nodes per class
        """

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


def load_dataset(data_dir, dataname, prefix, sub_dataname=''):
    if dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(data_dir, dataname)
    elif dataname in ('trojan'):
        dataset = load_trojan_data(data_dir,dataname,prefix)
    elif dataname in ('amazon-photo', 'amazon-computer'):
        dataset = load_amazon_dataset(data_dir, dataname)
    elif dataname in ('coauthor-cs', 'coauthor-physics'):
        dataset = load_coauthor_dataset(data_dir, dataname)
    elif dataname in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset = load_geom_gcn_dataset(data_dir, dataname)
    elif dataname == 'ogbn-proteins':
        dataset = load_proteins_dataset(data_dir)
    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname == 'amazon2m':
        dataset = load_amazon2m_dataset(data_dir)
    elif dataname == 'twitch-e':
        if sub_dataname not in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'):
            print('Invalid sub_dataname, deferring to DE graph')
            sub_dataname = 'DE'
        dataset = load_twitch_dataset(data_dir, sub_dataname)
    elif dataname == 'fb100':
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'):
            print('Invalid sub_dataname, deferring to Penn94 graph')
            sub_dataname = 'Penn94'
        dataset = load_fb100_dataset(data_dir, sub_dataname)
    elif dataname == 'deezer-europe':
        dataset = load_deezer_dataset(data_dir)
    elif dataname == 'arxiv-year':
        dataset = load_arxiv_year_dataset(data_dir)
    elif dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)
    elif dataname == 'snap-patents':
        dataset = load_snap_patents_mat(data_dir)
    elif dataname == 'yelp-chi':
        dataset = load_yelpchi_dataset(data_dir)
    elif dataname == 'mini':
        dataset =load_mini_imagenet(data_dir)
    elif dataname == '20news':
        dataset=load_20news(data_dir)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_twitch_dataset(data_dir, sub_dataset):
    assert sub_dataset in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'

    filepath = data_dir + f"twitch/{sub_dataset}"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{sub_dataset}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2] == "True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{sub_dataset}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{sub_dataset}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]

    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)),
                                 (np.array(src), np.array(targ))),
                                shape=(n, n))
    features = np.zeros((n, 3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    new_label = label[reorder_node_ids]
    label = new_label

    dataset = NCDataset(sub_dataset)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_fb100_dataset(data_dir, sub_dataset):
    feature_vals_all = np.empty((0, 6))
    for f in ['Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98']:
        mat = scipy.io.loadmat(data_dir + 'facebook100/' + f + '.mat')
        A = mat['A']
        metadata = mat['local_info']
        metadata = metadata.astype(np.int)
        feature_vals = np.hstack(
            (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        feature_vals_all = np.vstack(
            (feature_vals_all, feature_vals)
        )

    mat = scipy.io.loadmat(data_dir + 'facebook100/' + sub_dataset + '.mat')
    A = mat['A']
    metadata = mat['local_info']
    dataset = NCDataset(sub_dataset)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feature_vals_all[:, col]))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    dataset.label = torch.where(dataset.label > 0, 1, 0)
    return dataset


def load_deezer_dataset(data_dir):
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{data_dir}deezer/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_arxiv_year_dataset(data_dir, nclass=5):
    filename = 'arxiv-year'
    dataset = NCDataset(filename)
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    label = even_quantile_labels(
        dataset.graph['node_year'].flatten(), nclass, verbose=False)
    dataset.label = torch.as_tensor(label).reshape(-1, 1)
    return dataset


def load_proteins_dataset(data_dir):
    ogb_dataset = NodePropPredDataset(name='ogbn-proteins', root=f'{data_dir}/ogb')
    dataset = NCDataset('ogbn-proteins')
    def protein_orig_split(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        return {'train': torch.as_tensor(split_idx['train']),
                'valid': torch.as_tensor(split_idx['valid']),
                'test': torch.as_tensor(split_idx['test'])}
    dataset.load_fixed_splits = protein_orig_split
    dataset.graph, dataset.label = ogb_dataset.graph, ogb_dataset.labels

    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['edge_feat'] = torch.as_tensor(dataset.graph['edge_feat'])
    dataset.label = torch.as_tensor(dataset.label)

    edge_index_ = to_sparse_tensor(dataset.graph['edge_index'],
                                   dataset.graph['edge_feat'], dataset.graph['num_nodes'])
    dataset.graph['node_feat'] = edge_index_.mean(dim=1)
    dataset.graph['edge_feat'] = None
    return dataset

def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx

    dataset.load_fixed_splits = ogb_idx_to_tensor
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset

def load_amazon2m_dataset(data_dir):
    ogb_dataset = NodePropPredDataset(name='ogbn-products', root=f'{data_dir}/ogb')
    dataset = NCDataset('amazon2m')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)

    def load_fixed_splits(train_prop=0.5, val_prop=0.25):
        dir = f'{data_dir}ogb/ogbn_products/split/random_0.5_0.25'
        tensor_split_idx = {}
        if os.path.exists(dir):
            tensor_split_idx['train'] = torch.as_tensor(np.loadtxt(dir + '/amazon2m_train.txt'), dtype=torch.long)
            tensor_split_idx['valid'] = torch.as_tensor(np.loadtxt(dir + '/amazon2m_valid.txt'), dtype=torch.long)
            tensor_split_idx['test'] = torch.as_tensor(np.loadtxt(dir + '/amazon2m_test.txt'), dtype=torch.long)
        else:
            os.makedirs(dir)
            tensor_split_idx['train'], tensor_split_idx['valid'], tensor_split_idx['test'] \
                = rand_train_test_idx(dataset.label, train_prop=train_prop, valid_prop=val_prop)
            np.savetxt(dir + '/amazon2m_train.txt', tensor_split_idx['train'], fmt='%d')
            np.savetxt(dir + '/amazon2m_valid.txt', tensor_split_idx['valid'], fmt='%d')
            np.savetxt(dir + '/amazon2m_test.txt', tensor_split_idx['test'], fmt='%d')
        return tensor_split_idx
    dataset.load_fixed_splits = load_fixed_splits
    return dataset

def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['pokec'], \
            dest_path=f'{data_dir}pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_snap_patents_mat(data_dir, nclass=5):
    if not path.exists(f'{data_dir}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['snap-patents'], \
            dest_path=f'{data_dir}snap_patents.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}snap_patents.mat')

    dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_yelpchi_dataset(data_dir):
    if not path.exists(f'{data_dir}YelpChi.mat'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['yelp-chi'], \
            dest_path=f'{data_dir}YelpChi.mat', showsize=True)
    fulldata = scipy.io.loadmat(f'{data_dir}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset

torch_dataset_file = open("torch_dataset.txt","w")

def load_planetoid_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    torch_dataset = Planetoid(root=f'{data_dir}Planetoid',
                              name=name, transform=transform)
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    # print(edge_index)
    # edge_example = edge_index[:, np.where(edge_index[0]==30)[0]]
    # print(edge_example)
    # tensor([[  30,   30,   30,   30,   30,   30],
    #     [ 697,  738, 1358, 1416, 2162, 2343]])

    node_feat = data.x
    # print(node_feat)
    label = data.y
    print(len(label))
    num_nodes = data.num_nodes

    dataset = NCDataset(name)
    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    # print(node_feat)
    # print(num_nodes)
    return dataset


def load_trojan_data(data_dir,dataname,prefix):
    G_location = os.path.join(data_dir,prefix+"-G.json")
    G_data = json.load(open(G_location))

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
    feat_location = os.path.join(data_dir,prefix+"-feats.npy")
    if os.path.exists(feat_location):
        feats = np.load(feat_location)
    else:
        print("No features present.. Only identity features will be used.")
        feats = None    
    node_feat = torch.tensor(feats,dtype=torch.float32)  # Convert feats to torch.Tensor

    # num_nodes
    num_nodes = G.number_of_nodes()
    # val_mask
    val_nodes = []

    for node in G.nodes(data=True):
        if node[1].get('val', False):
            val_nodes.append(node[0])
    val_mask = torch.tensor(val_nodes)
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
    class_location = os.path.join(data_dir,prefix+"-class_map.json")
    class_map = json.load(open(class_location))
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



def load_amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Photo', transform=transform)
    elif name == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Computers', transform=transform)
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='CS', transform=transform)
    elif name == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='Physics', transform=transform)
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_geom_gcn_dataset(data_dir, name):
    graph_adjacency_list_file_path = f'{data_dir}geom-gcn/{name}/out1_graph_edges.txt'
    graph_node_features_and_labels_file_path = f'{data_dir}geom-gcn/{name}/out1_node_feature_label.txt'

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.tocoo().astype(np.float32)
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    def preprocess_features(feat):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(feat.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        feat = r_mat_inv.dot(feat)
        return feat

    features = preprocess_features(features)

    edge_index = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = labels

    return dataset


def load_mini_imagenet(data_dir):
    import pickle as pkl

    dataset = NCDataset('mini_imagenet')

    data = pkl.load(open(data_dir + 'mini_imagenet/mini_imagenet.pkl', 'rb'))
    x_train = data['x_train']
    x_val = data['x_val']
    x_test = data['x_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    features = torch.cat((x_train, x_val, x_test), dim=0)
    labels = np.concatenate((y_train, y_val, y_test))
    num_nodes = features.shape[0]

    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = torch.LongTensor(labels)
    return dataset


def load_20news(data_dir, n_remove=0):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    import pickle as pkl

    if path.exists(data_dir + '20news/20news.pkl'):
        data = pkl.load(open(data_dir + '20news/20news.pkl', 'rb'))
    else:
        categories = ['alt.atheism',
                      'comp.sys.ibm.pc.hardware',
                      'misc.forsale',
                      'rec.autos',
                      'rec.sport.hockey',
                      'sci.crypt',
                      'sci.electronics',
                      'sci.med',
                      'sci.space',
                      'talk.politics.guns']
        data = fetch_20newsgroups(subset='all', categories=categories)
        # with open(data_dir + '20news/20news.pkl', 'wb') as f:
        #     pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

    vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
    X_counts = vectorizer.fit_transform(data.data).toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    features = transformer.fit_transform(X_counts).todense()
    features = torch.Tensor(features)
    y = data.target
    y = torch.LongTensor(y)

    num_nodes = features.shape[0]

    if n_remove > 0:
        num_nodes-=n_remove
        features=features[:num_nodes,:]
        y=y[:num_nodes]

    dataset = NCDataset('20news')
    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = torch.LongTensor(y)

    return dataset



