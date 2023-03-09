# reproduce results on ogbn-proteins (the splits are provided by ogb paper)
python test_large_graph.py --dataset ogbn-proteins --metric rocauc --method nodeformer --num_layers 3 \
--hidden_channels 64 --num_heads 1 --rb_order 1 --rb_trans identity \
--lamda 0.1 --M 50 --K 5 --use_bn --use_residual --use_gumbel --use_act --use_jk

# reproduce results on amazon2m
# one needs to first download the fixed splits into '../data/ogb/ogbn_products/split/random_0.5_0.25'
python test_large_graph.py --dataset amazon2m --metric acc --method nodeformer --num_layers 3 \
--hidden_channels 64 --num_heads 1 --rb_order 1 --rb_trans identity \
--lamda 0.01 --M 50 --K 5 --use_bn --use_residual --use_gumbel --use_act --use_jk