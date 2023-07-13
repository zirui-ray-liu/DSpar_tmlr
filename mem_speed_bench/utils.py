
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from ogb.nodeproppred import Evaluator

def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask


ARXIV_EVAL = Evaluator(name='ogbn-arxiv')
PRODUCTS_EVAL = Evaluator(name='ogbn-products')

def ogb_get_acc(out, y, train_mask, val_mask, test_mask, dataset):
    y = y.unsqueeze(1)
    if dataset == 'arxiv':
        evaluator = ARXIV_EVAL
    elif dataset in ['products-cluster', 'products-saint']:
        evaluator = PRODUCTS_EVAL
    else:
        raise NotImplementedError
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': y[train_mask],
        'y_pred': y_pred[train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[val_mask],
        'y_pred': y_pred[val_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[test_mask],
        'y_pred': y_pred[test_mask],
    })['acc']

    return train_acc, valid_acc, test_acc


def get_optimizer(model_config, model):
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
    else:
        raise NotImplementedError
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'])
    else:
        raise NotImplementedError
    return optimizer
