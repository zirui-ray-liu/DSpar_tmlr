import sys
import os
import numpy as np
from torch.autograd import grad
sys.path.append(os.getcwd())
import argparse
import random
import time
import warnings
import yaml
import pdb

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.utils import subgraph
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from dspar import get_memory_usage, compute_tensor_bytes, exp_recorder
import models
from data import get_data
from logger import Logger
from sklearn.metrics import f1_score
import torch_geometric.transforms as T

MB = 1024**2
GB = 1024**3


parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True, 
                    help='the path to the configuration file')
parser.add_argument('--dataset', type=str, required=True, 
                    help='the name of the applied dataset')
parser.add_argument('--root', type=str, default='~/data')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--grad_norm', type=float, default=None)
parser.add_argument('--inductive', action='store_true')
parser.add_argument('--debug_mem', action='store_true')
parser.add_argument('--test_speed', action='store_true')
parser.add_argument('--amp', help='whether to enable apx mode', action='store_true')
parser.add_argument('--random_sparsify', help='whether to randomly sparsify the graph', action='store_true')
parser.add_argument('--spec_sparsify', help='whether to spectrally sparsify the graph', action='store_true')



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


def to_inductive(data):
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def train(model, optimizer, data, loss_op, grad_norm, scaler, amp_mode):
    model.train()
    optimizer.zero_grad()
    with autocast(enabled=amp_mode):
        out = model(data.x, data.adj_t)
        loss = loss_op(out[data.train_mask], data.y[data.train_mask])
    del data
    if amp_mode:
        scaler.scale(loss).backward()
        if grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
    return loss.item()


def compute_micro_f1(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
        
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.

@torch.no_grad()
def test(model, data, amp_mode):
    model.eval()
    with autocast(enabled=amp_mode):
        out = model(data.x, data.adj_t)
    y_true = data.y
    train_acc = compute_micro_f1(out, y_true, data.train_mask)
    valid_acc = compute_micro_f1(out, y_true, data.val_mask)
    test_acc = compute_micro_f1(out, y_true, data.test_mask)
    return train_acc, valid_acc, test_acc


def main():
    global args 
    args = parser.parse_args()
    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        name = model_config['name']
        loop = model_config.get('loop', False)
        normalize = model_config.get('norm', False)
        if args.dataset == 'reddit2':
            model_config = model_config['params']['reddit']
        else:
            model_config = model_config['params'][args.dataset]
        model_config['name'] = name
        model_config['loop'] = loop
        model_config['normalize'] = normalize

    print(f'model config: {model_config}')
    if args.dataset == 'yelp':
        multi_label = True
    else:
        multi_label = False
    print(f'clipping grad norm: {args.grad_norm}')
    args.model = model_config['arch_name']
    assert model_config['name'] in ['GCN', 'SAGE', 'GCN2']
    if args.amp:
        print('activate amp mode')
        scaler = GradScaler()
    else:
        scaler = None
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print("Use GPU {} for training".format(args.gpu))

    if args.spec_sparsify or args.random_sparsify:
        assert args.spec_sparsify ^ args.random_sparsify, "both the flags of random_sparsify and spec_sparsify are true."
        enable_sparsify = True
        suffix = 'mode: ' + 'random' if args.random_sparsify else 'spectral'
        print(f'enable sparsify flag, {suffix}')
    else:
        enable_sparsify = False
    torch.cuda.set_device(args.gpu)
    data, num_features, num_classes = get_data(args.root, args.dataset, False, enable_sparsify, args.random_sparsify)

    GNN = getattr(models, model_config['arch_name'])
    model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    loss_op = F.binary_cross_entropy_with_logits if multi_label else F.cross_entropy
    print(model)
    model.cuda(args.gpu)

    if args.debug_mem:
        print("========== Model and Optimizer only ===========")
        optimizer = get_optimizer(model_config, model)
        optimizer.zero_grad()
        model.reset_parameters()
        model.train()
        usage = get_memory_usage(args.gpu, False)
        exp_recorder.record("network", args.model)
        exp_recorder.record("model_only", usage / MB, 4)
        print("========== Load data to GPU ===========")
        print('converting data form...')
        s_time = time.time()
        data = T.ToSparseTensor()(data.to('cuda'))
        print(f'done. used {time.time() - s_time} sec')

        if model_config['loop']:
            t = time.perf_counter()
            print('Adding self-loops...', end=' ', flush=True)
            data.adj_t = data.adj_t.set_diag()
            print(f'Done! [{time.perf_counter() - t:.2f}s]')
        
        if model_config['normalize']:
            t = time.perf_counter()
            print('Normalizing data...', end=' ', flush=True)
            data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        if args.inductive:
            print('inductive learning mode')
            data = to_inductive(data)
        # data.adj_t.fill_cache_()
        init_mem = get_memory_usage(args.gpu, False)
        data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
        exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 4)
        out = model(data.x, data.adj_t)[data.train_mask]
        loss = loss_op(out, data.y[data.train_mask])
        print("========== Before Backward ===========")
        before_backward = get_memory_usage(args.gpu, True)
        act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss, out])

        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(res)

        loss.backward()
        optimizer.step()
        del loss, out
        print("========== After Backward ===========")
        after_backward = get_memory_usage(args.gpu, True)
        total_mem = before_backward + (after_backward - init_mem)
        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (total_mem / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(res)
        exp_recorder.record("total", total_mem / MB, 2)
        exp_recorder.record("activation", act_mem / MB, 2)
        # exp_recorder.dump('mem_results.json')
        s_time = time.time()
        if args.test_speed:
            model.reset_parameters()
            optimizer.zero_grad()
            epoch_per_sec = []
            for i in range(100):
                optimizer.zero_grad()
                t = time.time()
                torch.cuda.synchronize()
                out = model(data.x, data.adj_t)[data.train_mask]
                loss = loss_op(out, data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                duration = time.time() - t
                epoch_per_sec.append(duration)
                print(f'epoch {i}, duration: {duration} sec')
            print(f's/epoch: {np.mean(epoch_per_sec)}')
            print(f'training epoch/s: {100/np.sum(epoch_per_sec)}')

            model.eval()
            s_time = time.time()
            torch.cuda.synchronize()
            with torch.no_grad():
                for _ in range(100):
                    out = model(data.x, data.adj_t)           
            torch.cuda.synchronize()
            print(f'inference epoch/s: {100/(time.time() - s_time) }') 
        exit()

    print('converting data form...')
    s_time = time.time()
    data = T.ToSparseTensor()(data.to('cuda'))
    print(f'done. used {time.time() - s_time} sec')

    if model_config['loop']:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    if model_config['normalize']:
        t = time.perf_counter()
        print('Normalizing data...', end=' ', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    if args.inductive:
        print('inductive learning mode')
        data = to_inductive(data)
    logger = Logger(args.runs, args)
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        for epoch in range(1, 1 + model_config['epochs']):
            loss = train(model, optimizer, data, loss_op, args.grad_norm, scaler, args.amp)
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train Loss: {loss:.4f}')
    
            result = test(model, data, args.amp)
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train f1: {100 * train_acc:.2f}%, '
                    f'Valid f1: {100 * valid_acc:.2f}% '
                    f'Test f1: {100 * test_acc:.2f}%')

        logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()