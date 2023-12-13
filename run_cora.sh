python main.py --dataset cora --rand_split --metric acc --method nodeformer --lr 0.001 \
--weight_decay 5e-3 --num_layers 2 --hidden_channels 32 --num_heads 4 --rb_order 2 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 1 --epochs 5 --device 0
