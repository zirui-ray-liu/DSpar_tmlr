import sys
import os
import pdb
import numpy as np
import time
sys.path.append(os.getcwd())
import argparse
import scipy
import math

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch_geometric.transforms as T
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian
from data import get_data


EPSILON = 1 - math.log(2)
MB = 1024**2
GB = 1024**3


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='~/data')
parser.add_argument('--random_sparsify', help='whether to randomly sparsify the graph', action='store_true')
parser.add_argument('--spec_sparsify', help='whether to spectrally sparsify the graph', action='store_true')
parser.add_argument('--dataset', required=True)


def main():
    args = parser.parse_args()

    if args.spec_sparsify or args.random_sparsify:
        assert args.spec_sparsify ^ args.random_sparsify, "both the flags of random_sparsify and spec_sparsify are true."
        enable_sparsify = True
        suffix = 'mode: ' + 'random' if args.random_sparsify else 'spectral'
        print(f'enable sparsify flag, {suffix}')
    else:
        enable_sparsify = False
    data, num_features, num_classes = get_data(args.root, args.dataset, False, enable_sparsify, args.random_sparsify)
    lap_adj, lap_edge_weight = get_laplacian(data.edge_index, data.edge_attr, normalization='sym')
    sp_mat = to_scipy_sparse_matrix(lap_adj, lap_edge_weight, data.num_nodes)
    s_time = time.time()
    bottom_eigen = scipy.sparse.linalg.eigs(sp_mat, k=200, which='SR', return_eigenvectors=False).real
    top_eigen = scipy.sparse.linalg.eigs(sp_mat, k=200, which='LR', return_eigenvectors=False).real
    print(f'calcuate {args.dataset} eigen used {time.time() - s_time}')
    prefix = f'./{args.dataset}'
    if args.spec_sparsify:
        prefix += '_spec_spar'
    elif args.random_sparsify:
        prefix += '_random_spar'
    for suffix, arr in zip(['top', 'bottem'], [top_eigen, bottom_eigen]):
        fname = prefix + f'_{suffix}.npy'
        np.save(fname, arr)


if __name__ == '__main__':
    main()