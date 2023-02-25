# semi-supervised node classification with benchmark settings
python main.py --dataset cora --rand_split_class --metric acc --method nodeformer --lr 0.01 --dropout 0. \
--weight_decay 5e-4 --num_layers 2 --hidden_channels 32 --num_heads 4 --rb_order 3 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --seed 123 --device 3

python main.py --dataset citeseer --rand_split_class --metric acc --method nodeformer --lr 0.01 --dropout 0.5 \
--weight_decay 5e-3 --num_layers 2 --hidden_channels 32 --num_heads 4 --rb_order 3 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --seed 123 --device 3

python main.py --dataset pubmed --rand_split_class --metric acc --method nodeformer --lr 0.001 --dropout 0.3 \
--weight_decay 5e-4 --num_layers 2 --hidden_channels 32 --num_heads 4 --rb_order 3 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --seed 123 --device 3


# semi-supervised node classification with benchmark settings, fixed public split
python main.py --dataset cora --protocol semi --metric acc --method nodeformer --lr 0.01 --dropout 0. \
--weight_decay 5e-4 --num_layers 2 --hidden_channels 32 --num_heads 4 --rb_order 3 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --seed 42 --device 3

python main.py --dataset citeseer --protocol semi --metric acc --method nodeformer --lr 0.01 --dropout 0.5 \
--weight_decay 5e-3 --num_layers 2 --hidden_channels 32 --num_heads 4 --rb_order 3 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --seed 42 --device 3

python main.py --dataset pubmed --protocol semi --metric acc --method nodeformer --lr 0.001 --dropout 0.3 \
--weight_decay 5e-4 --num_layers 2 --hidden_channels 32 --num_heads 4 --rb_order 2 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --seed 42 --device 3


