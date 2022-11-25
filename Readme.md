The official implementation of NeurIPS22 Spotlight paper 
"NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification"

Related materials for our paper: 
[paper](https://www.researchgate.net/publication/363845271_NodeFormer_A_Scalable_Graph_Structure_Learning_Transformer_for_Node_Classification), 
[slides](https://qitianwu.github.io/assets/NodeFormer-slides.pdf), 
[blog in Chinese](),
[blog in English](),
[video in Chinese](https://www.bilibili.com/video/BV1MK411U7dc/?spm_id_from=333.788.recommend_more_video.2&vd_source=dd4795a9e34dbf19550fff1087216477),
[video in English]()

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

### Implementation Details

The following figure summaries how we achieve $O(N)$ complexity by means of organically
    combining Random Feature Map and Gumbel-Softmax strategies.

![](https://files.mdnice.com/user/23982/07be83d0-faca-4989-aebf-d913cd398070.png)

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

The datasets we used span three categories (Sec 4.1, 4.2, 4.3 in our paper)

- Transductive Node Classification: we use two homophilous graphs Cora and Citeseer and two heterophilic graphs Deezer-Europe and Actor.
These graph datasets are all public available at [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). The Deezer dataset is provided from 
[Non-Homophilous Benchmark](https://github.com/CUAI/Non-Homophily-Benchmarks),
and the Actor (also called Film) dataset is provided by [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn). 

- Large Graph Datasets: we use OGBN-Proteins and Amazon2M as two large-scale datasets. These datasets are available
at [OGB](https://github.com/snap-stanford/ogb). The original Amazon2M is collected by [ClusterGCN](https://arxiv.org/abs/1905.07953) and 
later used to construct the OGBN-Products.

- Graph-Enhanced Classification: we also consider two datasets without input graphs, i.e., Mini-Imagenet and 20News-Group 
for image and text classification, respectively. The Mini-Imagenet dataset is provided by [Matching Network](https://arxiv.org/abs/1606.04080),
and 20News-Group is available at [Scikit-Learn](https://jmlr.org/papers/v12/pedregosa11a.html)

We also provide an easy access to common datasets including what we used in the Google drive:

    https://drive.google.com/drive/folders/1sWIlpeT_TaZstNB5MWrXgLmh522kx4XV?usp=share_link

### How to run our codes?

1. Install the required package according `requirements.txt`

2. Download the datasets into a folder `../data`

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

### TODO

- [x] Release the codes and model implementation.

- [ ] Provide a detailed tutorial for NodeFormer.

- [ ] Provide an example demo for emb/structure visualization.

### Further Discussions

Our work takes an initial step for exploring how to build a scalable graph Transformer model
for node classification, and we also believe there exists ample room for further research and development
as future works. One can also use our implementation `kernelized_softmax()` and `kernelized_gumbel_softmax()`
for related projects concerning e.g., structure learning and communication, where the scalability matters.

### Citation

If you find this repo and our codes helpful, please consider citing our work

```bibtex
      @inproceedings{wu2022nodeformer,
      title = {NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification},
      author = {Qitian Wu and Wentao Zhao and Zenan Li and David Wipf and Junchi Yan},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      year = {2022}
      }
```

### ACK

We refer to the [Performer](https://github.com/lucidrains/performer-pytorch) work for the implementation of softmax kernel computation, 
and the pipeline for training and preprocessing is developed on basis of the [Non-Homophilous Benchmark](https://github.com/CUAI/Non-Homophily-Benchmarks) project. 