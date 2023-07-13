from typing import Tuple

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import (Yelp, Flickr, Reddit2, Reddit)
from torch_geometric.utils import to_undirected, remove_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
from utils import index2mask
from dspar.sparsification import maybe_sparsfication
from torch_sparse import SparseTensor
from torch_geometric.utils import scatter


def build_adj_t(data):
    (row, col), N = data.edge_index, data.num_nodes
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]
    value = None
    for key in ['edge_weight', 'edge_attr', 'edge_type']:
        if data[key] is not None:
            value = data[key][perm]
            break
    adj_t = SparseTensor(row=col, col=row, value=value,
                                  sparse_sizes=(N, N), is_sorted=True)
    adj_t.storage.rowptr()
    adj_t.storage.csc()
    return adj_t


def get_arxiv(root: str, enable_sparsify: bool, random_sparsify: bool, follow_by_subgraph_sampling: bool) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB')
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    if enable_sparsify:
        data = maybe_sparsfication(data, 'ogbn-arxiv', follow_by_subgraph_sampling, random_sparsify)
    data.x = data.x.contiguous()
    data.node_year = None
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    data.train_idx = split_idx['train']
    return data, dataset.num_features, dataset.num_classes


def get_products(root: str, enable_sparsify: bool, random_sparsify: bool, follow_by_subgraph_sampling: bool) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB')
    data = dataset[0]
    if enable_sparsify:
        data = maybe_sparsfication(data, 'ogbn-products', follow_by_subgraph_sampling, random_sparsify, is_undirected=False)
    data.x = data.x.contiguous()
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_yelp(root: str, enable_sparsify: bool, random_sparsify: bool, follow_by_subgraph_sampling: bool) -> Tuple[Data, int, int]:
    dataset = Yelp(f'{root}/YELP')
    data = dataset[0]
    if enable_sparsify:
        data = maybe_sparsfication(data, 'yelp', follow_by_subgraph_sampling, random_sparsify)
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_flickr(root: str, enable_sparsify: bool, random_sparsify: bool, follow_by_subgraph_sampling: bool) -> Tuple[Data, int, int]:
    dataset = Flickr(f'{root}/Flickr')
    data = dataset[0]
    if enable_sparsify:
        data = maybe_sparsfication(data, 'flickr', follow_by_subgraph_sampling, random_sparsify)
    return data, dataset.num_features, dataset.num_classes


def get_reddit(root: str, enable_sparsify: bool, random_sparsify: bool, follow_by_subgraph_sampling: bool) -> Tuple[Data, int, int]:
    dataset = Reddit(f'{root}/Reddit')
    data = dataset[0]
    if enable_sparsify:
        data = maybe_sparsfication(data, 'reddit', follow_by_subgraph_sampling, random_sparsify)
    return data, dataset.num_features, dataset.num_classes


def get_reddit2(root: str, enable_sparsify: bool, random_sparsify: bool, follow_by_subgraph_sampling: bool) -> Tuple[Data, int, int]:
    dataset = Reddit2(f'{root}/Reddit2')
    data = dataset[0]
    if enable_sparsify:
        data = maybe_sparsfication(data, 'reddit2', follow_by_subgraph_sampling, random_sparsify)
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_proteins(root: str, enable_sparsify: bool, random_sparsify: bool, follow_by_subgraph_sampling: bool) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-proteins', f'{root}/OGB', transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    col = data.adj_t.storage.col()
    data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum') # add node features from edge features
    data.adj_t = None
    if enable_sparsify:
        data = maybe_sparsfication(data, 'ogbn-proteins', follow_by_subgraph_sampling, random_sparsify)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, data.num_features, 112 # the number of node classes, from ogb examples


def get_data(root: str, name: str, follow_by_subgraph_sampling: bool=False, enable_sparsify: bool=False, random_sparsify: bool=True) -> Tuple[Data, int, int]:
    if name.lower() == 'reddit':
        return get_reddit(root, enable_sparsify, random_sparsify, follow_by_subgraph_sampling)
    elif name.lower() == 'reddit2':
        return get_reddit2(root, enable_sparsify, random_sparsify, follow_by_subgraph_sampling)
    elif name.lower() == 'flickr':
        return get_flickr(root, enable_sparsify, random_sparsify, follow_by_subgraph_sampling)
    elif name.lower() == 'yelp':
        return get_yelp(root, enable_sparsify, random_sparsify, follow_by_subgraph_sampling)
    elif name.lower() == 'arxiv':
        return get_arxiv(root, enable_sparsify, random_sparsify, follow_by_subgraph_sampling)
    elif name.lower() == 'proteins':
        return get_proteins(root, enable_sparsify, random_sparsify, follow_by_subgraph_sampling)
    elif name.lower() == 'products':
        return get_products(root, enable_sparsify, random_sparsify, follow_by_subgraph_sampling)
    else:
        raise NotImplementedError