import pdb
import time
import random
random.seed(42)
import numpy as np
import torch
import dspar.cpp_extension.sampler as sampler
from torch_geometric.utils import degree, to_undirected


def maybe_sparsfication(data, dataset, follow_by_subgraph_sampling, random=False, is_undirected=True, reweighted=True):
    N, E = data.num_nodes, data.num_edges
    src, dst = data.edge_index
    if dataset == 'ogbn-arxiv':
        epsilon = 0.25 if not random else 0.35
    elif dataset == 'reddit2':
        epsilon = 0.3 if not random else 0.32
    elif dataset == 'ogbn-products':
        epsilon = 0.4 if not random else 0.45
    elif dataset == 'yelp':
        epsilon = 0.5 if not random else 0.6
    elif dataset == 'ogbn-proteins':
        epsilon = 0.25

    if follow_by_subgraph_sampling and dataset == 'ogbn-products':
        epsilon = 0.15 if not random else 0.2

    print(f'epsilon: {epsilon}')
    Q = int(0.16 * N * np.log(N) / epsilon ** 2)
    print(f"Q: {Q}")
    print(f'E/Q ratio: {E/Q}')
    print(f'E/nlogn ratio: {E/N/np.log(N)}')
    print('sparsify the input graph')
    data = data.clone()
    s = time.time()
    if random:
        pe = torch.ones(size=(E,), dtype=torch.double) / E
    else:
        print('sparsify the graph by degrees')
        node_degree = degree(dst, data.num_nodes)
        di, dj = torch.nan_to_num(1. / node_degree[src]), torch.nan_to_num(1. / node_degree[dst])
        pe = (di + dj).double()
        pe = pe / torch.sum(pe)
    p_cumsum = torch.cumsum(pe, 0)
    print(f'cal edge distribution used {time.time() - s} sec')
    # For reproducibility, we manually set the seed of graph sparsification to 42. We note that this seed is only effective for the graph sparsification, 
    # it does not impact any following process.
    seed_val = 42
    s = time.time()
    sampled = sampler.edge_sample(p_cumsum, Q, seed_val)
    print(f'sample edge used {time.time() - s} sec')
    e_indices, e_cnt = torch.unique(sampled, return_counts=True)
    new_graph = e_cnt / Q / pe[e_indices]
    new_src, new_dst = src[e_indices], dst[e_indices]
    edge_index = torch.cat([new_src.view(1, -1), new_dst.view(1, -1)], dim=0)
    edge_attr = new_graph.float()
    if is_undirected:
        data.edge_index, data.edge_attr = to_undirected(edge_index, edge_attr)
    else:
        data.edge_index, data.edge_attr = edge_index, edge_attr 
    if not reweighted:
        print('not reweight')
        data.edge_attr = None
    print(f'before sparsification, num_edges: {E}, after sparsification, num_edges: {new_src.shape[0]}, ratio: {new_src.shape[0] / E}')
    return data