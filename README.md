This is the official codes for DSpar: Embarrassingly Simple Strategy for Efficient GNN training and inference via Degree-based Sparsification.

## Install
This code is tested with Python 3.8 and CUDA 11.0. To reproduce the results in this paper, please follow the below configuration.


- Create and activate conda environment.

<!-- ```
torch == 1.9.0
torch_geometric == 1.7.2
torch_scatter == 2.0.8
torch_sparse == 0.6.12
``` -->

```
conda env create -f environment.yml
conda activate graph
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```


- Build
```bash
cd src
pip install -v -e .
```

## Reproduce results

### Reproduce {Reddit, YELP} results.
```bash
cd mem_speed_bench
python ./non_ogb_datasets/train_full_batch.py --conf ./non_ogb_datasets/conf/$MODEL.yaml --$SPARSIFY --dataset $DATASET
```
MODEL must be chosen from {gcn, sage, gcn2}, SPARSIFY can be chosen from {spec_sparsify, random_sparsify}. spec_sparsity corresponds to Algorithm 1 in our paper. DATASET must be chosen from {reddit2, yelp}


If you do not want to apply any graph sparsification, just remove ```--$SPARSIFY```

For mini-batch training, 
```bash
cd mem_speed_bench
python ./non_ogbn_datasets/train_mini_batch.py --conf ./non_ogbn_datasets/conf/$MODEL.yaml --$SPARSIFY --grad_norm $GRAD_NORM
```
MODEL must be chosen from {saint_sage, cluster_gcn}. 
SPARSIFY can be chosen from {spec_sparsify, random_sparsify}.
DATASET must be chosen from {reddit2, yelp}.
For GRAD_NORM, it can found in appendix.

### Reproduce ogbn-arxiv results.
```bash
cd mem_speed_bench
python ./arxiv/train_full_batch.py --conf ./arxiv/conf/$MODEL.yaml --$SPARSIFY
```
MODEL must be chosen from {gcn, sage, gcn2}, SPARSIFY can be chosen from {spec_sparsify, random_sparsify}. spec_sparsity corresponds to Algorithm 1 in our paper.


### Reproduce ogbn-products results.
For full-batch training, 
```bash
cd mem_speed_bench
python ./products/train_full_batch.py --conf ./products/conf/sage.yaml --$SPARSIFY
```
SPARSIFY can be chosen from {spec_sparsify, random_sparsify}.

For mini-batch training, 
```bash
cd mem_speed_bench
python ./yaml/train_mini_batch.py --conf ./yaml/conf/$MODEL.yaml --$SPARSIFY
```
MODEL must be chosen from {cluster_sage, saint_sage}.
SPARSIFY can be chosen from {spec_sparsify, random_sparsify}.


### Get the occupied memory and training throughout.
Add the flag **--deug_mem** and **--test_speed** to the above commends. For example,
```
python ./arxiv/train_full_batch.py --conf ./arxiv/conf/$MODEL.yaml --n_bits $BIT_WIDTH $SPARSIFY --debug_mem --test_speed
```

### Combining dspar and AMP
Add the flag **--amp** to the above commends.

## Acknowledgment
Our code is based on the official code of [GNNAutoScale](https://arxiv.org/abs/2106.05609).