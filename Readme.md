## NodeFormer: A Graph Transformer for Node-Level Prediction
 
The official implementation for "NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification" which
is accepted to NeurIPS22 as a spotlight presentation.

Related materials: 
[paper](https://www.researchgate.net/publication/363845271_NodeFormer_A_Scalable_Graph_Structure_Learning_Transformer_for_Node_Classification), 
[slides](https://qitianwu.github.io/assets/NodeFormer-slides.pdf), 
[blog in Chinese](https://zhuanlan.zhihu.com/p/587086593),
[blog in English](https://towardsdatascience.com/scalable-graph-transformers-for-million-nodes-2f0014ceb9d4)

## What's news

- [2022.11.27] We release the early version of codes for reproducibility.

- [2023.02.20] We provide the checkpoints of NodeFormer on ogbn-Proteins and Amazon2M (see [here](https://github.com/qitianwu/NodeFormer#how-to-run-our-codes) for details).

- [2023.03.08] We add results on cora, citeseer, pubmed with semi-supervised random splits (see [here](https://github.com/qitianwu/NodeFormer#key-results) for details)

This work takes an initial step for exploring Transformer-style graph encoder networks for 
large node classification graphs, dubbed as ***NodeFormer***, as an
alternative to common Graph Neural Networks, in particular for encoding nodes in an
input graph into embeddings in latent space. 

![](https://files.mdnice.com/user/23982/7343df02-05cb-43fc-a3e3-ca7b4434a5d8.png)

### The highlights of ***NodeFormer***

NodeFormer is a pioneering Transformer model for node classification on large graphs. NodeFormer scales 
all-pair message passing with efficient latent structure learning to million-level nodes. NodeFormer has several merits:

- **All-Pair Message Passing on Layer-specific Adaptive Structures**. The feature propagation per layer 
    is operated on a latent graph that potentially connect all the nodes, in contrast with the local
    propagation design of GNNs that only aggregates the embeddings of neighbored nodes.

- **Linear Complexity w.r.t. Node Numbers**. The all-pair message passing on latent graphs that are optimized 
    together only requires $O(N)$ complexity, empowered by our proposed ***kernelized Gumbel-Softmax operator***. The largest demonstration of our model in our paper is the graph with 
    2M nodes, yet we believe it can even scale to larger ones with the mini-batch partition.

- **Efficient End-to-End Learning for Latent Structures**. The optimization for the latent structures is allowed
for end-to-end training with the model, making the whole learning process simple and efficient. E.g., the training on Cora only requires 1-2 minutes, 
while on OGBN-Proteins requires 1-2 hours in one run.

- **Flexibility for Inductive Learning and Graph-Free Scenarios**. NodeFormer is flexible for handling new unseen nodes in testing and 
as well as predictive tasks without input graphs, e.g., image and text classification. It can also be used for interpretability analysis
with the latent interactions among data points explicitly estimated.

### Structures of the Codes

The key module of NodeFormer is the ***kernelized (Gumbel-)Softmax message passing*** which achieves all-pair message passing on a latent
graph in each layer with $O(N)$ complexity. The `nodeformer.py` implements our model:

- The functions `kernelized_softmax()` and `kernelized_gumbel_softmax()` implement the message passing per layer. The Gumbel version
is only used for training.

- The layer class `NodeFormerConv` implements one-layer feed-forward of NodeFormer (which contains MP on a latent graph, 
adding relational bias and computing edge-level reg loss from input graphs if available).

- The model class `NodeFormer` implements the model that adopts standard input (node features, adjacency) and output 
(node prediction, edge loss).

For other files, the descriptions are below:

- `main.py` is the pipeline for full-graph training/evaluation.

- `main-batch.py` is the pipeline for training with random mini-batch partition for large datasets.

### Datasets

We provide an easy access to the used datasets in the [Google drive](https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=sharing).
This also contains other commonly used graph datasets, except the large-scale graphs OGBN-Proteins and Amazon2M which can be downloaded automatically with our [codes](https://github.com/qitianwu/NodeFormer/blob/50fe19b7b1e67290d42d5b98ae82a981be40631c/dataset.py#L267)
See [here](https://github.com/qitianwu/NodeFormer#how-to-run-our-codes) for how to get the datasets ready for running our codes.

The information and sources of datasets are summarized below

- Transductive Node Classification (Sec 4.1 in paper): we use two homophilous graphs Cora and Citeseer and two heterophilic graphs Deezer-Europe and Actor.
These graph datasets are all public available at [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). The Deezer dataset is provided from 
[Non-Homophilous Benchmark](https://github.com/CUAI/Non-Homophily-Benchmarks),
and the Actor (also called Film) dataset is provided by [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn). 

- Large Graph Datasets (Sec 4.2 in paper): we use OGBN-Proteins and Amazon2M as two large-scale datasets. These datasets are available
at [OGB](https://github.com/snap-stanford/ogb). The original Amazon2M is collected by [ClusterGCN](https://arxiv.org/abs/1905.07953) and 
later used to construct the OGBN-Products.

- Graph-Enhanced Classification (Sec 4.3 in paper): we also consider two datasets without input graphs, i.e., Mini-Imagenet and 20News-Group 
for image and text classification, respectively. The Mini-Imagenet dataset is provided by [Matching Network](https://arxiv.org/abs/1606.04080),
and 20News-Group is available at [Scikit-Learn](https://jmlr.org/papers/v12/pedregosa11a.html)



### Key results

| Dataset | Split | Metric | Result | Hyper-parameters/Checkpoints |
| --- | --- | --- | --- | --- |
| Cora | random 50%/25%/25% | Accuracy | 88.80 (0.26) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run.sh) | 
| CiteSeer | random 50%/25%/25% | Accuracy | 76.33 (0.59) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run.sh) |
| Deezer | random 50%/25%/25% | ROC-AUC | 71.24 (0.32) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run.sh) |
| Actor | random 50%/25%/25% | Accuracy | 35.31 (0.89) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run.sh) |
| OGBN-Proteins | public split | ROC-AUC | 77.45 (1.15) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run.sh), [checkpoint](https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=sharing), [test script](https://github.com/qitianwu/NodeFormer/blob/main/run_test_large_graph.sh) |
| Amazon2M | random 50%/25%/25% | Accuracy | 87.85 (0.24) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run.sh), [checkpoint and fixed splits](https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=sharing), [test script](https://github.com/qitianwu/NodeFormer/blob/main/run_test_large_graph.sh) |
| Mini-ImageNet (kNN, k=5) | random 50%/25%/25% | Accuracy | 86.77 (0.45) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run.sh) |
| Mini-ImageNet (no graph) | random 50%/25%/25% | Accuracy | 87.46 (0.36) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run.sh) |
| 20News-Group (kNN, k=5) | random 50%/25%/25% | Accuracy | 66.01 (1.18) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run.sh) |
| 20News-Group (no graph) | random 50%/25%/25% | Accuracy | 64.71 (1.33) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run.sh) |
| Cora | 20 nodes per class for train | Accuracy |  83.4 (0.2) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run_extra.sh) | 
| CiteSeer | 20 nodes per class for train | Accuracy | 73.0 (0.3) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run_extra.sh) |
| Pubmed | 20 nodes per class for train | Accuracy | 81.5 (0.4) | [train script](https://github.com/qitianwu/NodeFormer/blob/main/run_extra.sh) |


### How to run our codes?

1. Install the required package according to `requirements.txt`

2. Create a folder `../data` and download the datasets from [here](https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=sharing)
(For large graph datasets Proteins and Amazon2M, the datasets will be automatically downloaded)

3. To run the training and evaluation on eight datasets we used, one can use the scripts in `run.sh`:

```shell
# node classification on small datasets
python main.py --dataset cora --rand_split --metric acc --method nodeformer --lr 0.001 \
--weight_decay 5e-3 --num_layers 2 --hidden_channels 32 --num_heads 4 --rb_order 2 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --device 0

# node classification on large datasets
python main-batch.py --dataset ogbn-proteins --metric rocauc --method nodeformer --lr 1e-2 \
--weight_decay 0. --num_layers 3 --hidden_channels 64 --num_heads 1 --rb_order 1 --rb_trans identity \
--lamda 0.1 --M 50 --K 5 --use_bn --use_residual --use_gumbel --use_act --use_jk --batch_size 10000 \
--runs 5 --epochs 1000 --eval_step 9 --device 0

# image and text datasets
python main.py --dataset mini --metric acc --rand_split --method nodeformer --lr 0.001\
--weight_decay 5e-3 --num_layers 2 --hidden_channels 128 --num_heads 6\
--rb_order 2 --rb_trans sigmoid --lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel \
--run 5 --epochs 300 --device 0

```

4. We also provide the [checkpoints](https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=sharing) of NodeFormer on two large datasets, OGBN-Proteins and Amazon2M.
One can download the trained models into `../model/` and run the scripts in `run_test_large_graph.sh` for reproducing the results. 

- For Amazon2M, to ensure obtaining the result as ours, one need to download the fixed splits from [here](https://drive.google.com/drive/folders/1zt6R05fvXfi-sWrLtxj-Pe3OgIGSWqFf?usp=sharing) to `../data/ogb/ogbn_products/split/random_0.5_0.25`.

### Potential Applications and More Usability

NodeFormer can in principle be applied to solve three families of tasks:

- ***Node-Level Prediction on (Large) Graphs***: use NodeFormer to replace GNN encoder as an encoder backbone for graph-structured data.

- ***General Machine Learning Problems***: use NodeFormer as an encoder that computes instance representations by their global all-pair interactions, for general ML tasks, e.g., classification or regression.

- ***Learning Latent Graph Structures***: use NodeFormer to learn latent graphs among a set of objects.

Our work takes an initial step for exploring how to build a scalable graph Transformer model
for node classification, and we also believe there exists ample room for further research and development
as future works. One can also use our implementation `kernelized_softmax()` and `kernelized_gumbel_softmax()`
for related projects concerning e.g., structure learning and communication, where the scalability matters.

### Citation

If you find our codes useful, please consider citing our work

```bibtex
      @inproceedings{wu2022nodeformer,
      title = {NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification},
      author = {Qitian Wu and Wentao Zhao and Zenan Li and David Wipf and Junchi Yan},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      year = {2022}
      }
```

### ACK

We acknowledge the implementation of the softmax kernel 

https://github.com/lucidrains/performer-pytorch

and the training pipeline for GNN node classification
 
https://github.com/CUAI/Non-Homophily-Benchmarks 