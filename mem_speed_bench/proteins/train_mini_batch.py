import sys
import os
import pdb
import numpy as np
sys.path.append(os.getcwd())
import argparse
import random
import time
import warnings
import yaml

import math
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import sum as sparsesum, mul
from torch_geometric.data import GraphSAINTRandomWalkSampler, ClusterData, ClusterLoader

from dspar import get_memory_usage, compute_tensor_bytes, exp_recorder
import models
from data import get_data
from logger import Logger
from sklearn.metrics import f1_score
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator

MB = 1024**2
GB = 1024**3

SAMPLER = {'graph_saint': GraphSAINTRandomWalkSampler, 'cluster_gcn': ClusterData}

parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, required=True)
parser.add_argument('--root', type=str, default='~/data')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--efficient_eval', action='store_true', default=False, help='while set to True, we use larger eval_iter in the frist 800 epochs')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--debug_mem', action='store_true',
                    help='whether to debug the memory usage')
parser.add_argument('--n_bits', type=float, default=None)
parser.add_argument('--grad_norm', type=int, default=None)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--simulate', action='store_true')
parser.add_argument('--act_fp', action='store_true')
parser.add_argument('--kept_frac', type=float, default=1.0)
parser.add_argument('--amp', help='whether to enable apx mode', action='store_true')
parser.add_argument('--test_speed', action='store_true', help='whether to test the speed and throughout')
parser.add_argument('--random_sparsify', help='whether to randomly sparsify the graph', action='store_true')
parser.add_argument('--spec_sparsify', help='whether to spectrally sparsify the graph', action='store_true')
parser.add_argument('--eval_iter', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=12)


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


def train(model, optimizer, loader, loss_op, grad_norm, scaler, amp_mode):
    model.train()
    total_loss = 0

    for data in loader:
        # s_time = time.time()
        data = T.ToSparseTensor()(data.to('cuda'))
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)
        loss = loss_op(out[data.train_mask], data.y[data.train_mask].to(torch.float))
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        total_loss += loss.item()
        # print(f'used time: {time.time() - s_time}')
    return total_loss / len(loader)


@torch.no_grad()
@torch.no_grad()
def test(model, data, evaluator, amp_mode=False):
    model.eval()

    with autocast(enabled=amp_mode):
        y_pred = model(data.x, data.adj_t)

    #pdb.set_trace()
    # y = data.y.view(-1, 1)
    
    train_rocauc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': y_pred[data.train_mask],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[data.val_mask],
        'y_pred': y_pred[data.val_mask],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': y_pred[data.test_mask],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def preprocess_data(model_config, data):
    loop, normalize = model_config['loop'], model_config['normalize']
    #pdb.set_trace()
    if loop:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    if normalize:
        t = time.perf_counter()
        print('Normalizing data...', end=' ', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')


def main():
    args = parser.parse_args()
    with open(args.conf, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
    args.model = model_config['name'] # get the model name from the conf file
    assert args.model.lower() in ['graph_saint'] # list of full-batch training models
    print(args)
    print(model_config)
    
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
        print(f'Using GPU: {args.gpu} for training')
        torch.cuda.set_device(args.gpu)

    if args.spec_sparsify or args.random_sparsify:
        assert args.spec_sparsify ^ args.random_sparsify, "both the flags of random_sparsify and spec_sparsify are true."
        enable_sparsify = True
        suffix = 'mode: ' + 'random' if args.random_sparsify else 'spectral'
        print(f'enable sparsify flag, {suffix}')
    else:
        enable_sparsify = False

    if args.amp:
        print(f'amp mode: {args.amp}')


    print('use BCE loss with logits, bcz the dataset has multi-label')
    loss_op = torch.nn.BCEWithLogitsLoss()

    grad_norm = args.grad_norm
    print(f'clipping grad norm: {grad_norm}')

    data, num_features, num_classes = get_data(args.root, 'proteins', False, enable_sparsify, args.random_sparsify)
    sampler_data = data
    print('converting data form...')
    s_time = time.time()
    data = T.ToSparseTensor()(data.clone())
    data = data.to('cuda')
    # preprocess_data(model_config, data)
    print(f'done. used {time.time() - s_time} sec')

    sampler_cls = SAMPLER[model_config['name']]
    s_time = time.time()
    print("=" * 30 + f'Prepare {sampler_cls.__name__} for training' + '=' * 30)
    if model_config['name'] == 'graph_saint':
        num_steps = int(sampler_data.num_nodes / model_config['sampler']['batch_size'] / model_config['sampler']['walk_length'])
        print(f'num steps: {num_steps}')
        loader = sampler_cls(sampler_data,
                            num_workers=args.num_workers,
                            num_steps=num_steps,
                            **model_config['sampler'])

    elif model_config['name'] == 'cluster_gcn':
        batch_size = model_config['sampler']['batch_size']
        del model_config['sampler']['batch_size']
        cluster_data = sampler_cls(sampler_data, 
                                   **model_config['sampler'])
        loader = ClusterLoader(cluster_data, batch_size=batch_size, 
                               shuffle=True, num_workers=args.num_workers)
    else:
        raise NotImplementedError
    print("=" * 30 + 
          f'Finished Building {sampler_cls.__name__}, used {time.time() - s_time} sec' + 
          '=' * 30)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)
    GNN = getattr(models, model_config['arch_name'])
    model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture'])
    print(f'Model: {model}')
    model.cuda(args.gpu)

    if args.debug_mem:
        print("========== Model Only ===========")
        usage = get_memory_usage(args.gpu, True)
        exp_recorder.record("network", 'GCN')
        exp_recorder.record("model_only", usage / MB, 2)
        print("========== Load data to GPU ===========")
        data.adj_t.fill_cache_()
        data = data.to('cuda')
        init_mem = get_memory_usage(args.gpu, True)
        data_mem = init_mem / MB - exp_recorder.val_dict['model_only']
        exp_recorder.record("data", init_mem / MB - exp_recorder.val_dict['model_only'], 2)
        model.reset_parameters()
        model.train()
        optimizer = get_optimizer(model_config, model)
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)[data.train_mask]
        loss = loss_op(out, data.y.squeeze(1)[data.train_mask].to(torch.float))
        print(f'max allocated mem (MB): {torch.cuda.max_memory_allocated(0) / MB}')
        print("========== Before Backward ===========")
        del out
        before_backward = get_memory_usage(args.gpu, True)
        act_mem = get_memory_usage(args.gpu, False) - init_mem - compute_tensor_bytes([loss])
        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (before_backward / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(res) 
        loss.backward()
        optimizer.step()
        del loss
        print("========== After Backward ===========")
        after_backward = get_memory_usage(args.gpu, True)
        total_mem = before_backward + (after_backward - init_mem)
        res = "Total Mem: %.2f MB\tData Mem: %.2f MB\tAct Mem: %.2f MB" % (total_mem / MB,
                                                                           data_mem,
                                                                           act_mem / MB)
        print(f'max allocated mem (MB): {torch.cuda.max_memory_allocated(0) / MB}')
        print(res)
        exp_recorder.record("total", total_mem / MB, 2)
        exp_recorder.record("activation", act_mem / MB, 2)
        exp_recorder.dump('mem_results.json')
        s_time = time.time()
        torch.cuda.synchronize()
        if args.test_speed:
            model.reset_parameters()
            optimizer.zero_grad()
            epoch_per_sec = []

            for i in range(100):
                t = time.time()
                optimizer.zero_grad()
                out = model(data.x, data.adj_t)[data.train_mask]
                loss = loss_op(out, data.y.squeeze(1)[data.train_mask].to(torch.float))
                loss.backward()
                optimizer.step()
                duration = time.time() - t
                epoch_per_sec.append(duration)
                print(f'epoch {i}, duration: {duration} sec')
            torch.cuda.synchronize()
            print(f'training epoch/s: {100/(time.time() - s_time) }')

            model.eval()
            s_time = time.time()
            torch.cuda.synchronize()
            with torch.no_grad():
                for _ in range(100):
                    out = model(data.x, data.adj_t)           
            torch.cuda.synchronize()
            print(f'inference epoch/s: {100/(time.time() - s_time) }') 
        exit()


    for run in range(args.runs):
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        if args.amp:
            print('activate amp mode')
            scaler = GradScaler()
        else:
            scaler = None
        durations = []
        for epoch in range(1, 1 + model_config['epochs']):
            s_time = time.time()
            torch.cuda.synchronize()
            loss = train(model, optimizer, loader, loss_op, grad_norm, scaler, args.amp)
            duration = time.time() - s_time
            durations.append(duration)
            # =========================== Validation ===========================

            if epoch % args.eval_iter == 0:
                result = test(model, data, evaluator, args.amp)
                logger.add_result(run, result)
                if  model_config['log_steps'] > 0 and epoch % model_config['log_steps'] == 0:
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')

        print(f'total training time: {np.sum(durations)}')
        logger.print_statistics(run)
    logger.print_statistics()

if __name__ == '__main__':
    main()