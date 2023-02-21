# node classification on small datasets
python main.py --dataset cora --rand_split --metric acc --method nodeformer --lr 0.001 \
--weight_decay 5e-3 --num_layers 2 --hidden_channels 32 --num_heads 4 --rb_order 2 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --device 3

python main.py --dataset citeseer --rand_split --metric acc --method nodeformer --lr 0.001 \
--weight_decay 5e-3 --num_layers 2 --hidden_channels 32 --num_heads 2 --rb_order 2 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --device 2

python main.py --dataset deezer-europe --rand_split --metric rocauc --method nodeformer --lr 1e-5 \
--weight_decay 5e-2 --num_layers 2 --num_heads 1 --rb_order 2 --rb_trans sigmoid --lamda 0.01 \
--M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --device 1

python main.py --dataset film --rand_split --metric acc --method nodeformer --lr 0.0001 \
--weight_decay 5e-2 --num_layers 2 --num_heads 1 --rb_order 2 --rb_trans sigmoid --lamda 0.01 \
--M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --device 0


# node classification on large datasets
python main-batch.py --dataset ogbn-proteins --metric rocauc --method nodeformer --lr 1e-2 \
--weight_decay 0. --num_layers 3 --hidden_channels 64 --num_heads 1 --rb_order 1 --rb_trans identity \
--lamda 0.1 --M 50 --K 5 --use_bn --use_residual --use_gumbel --use_act --use_jk --batch_size 10000 \
--runs 5 --epochs 1000 --eval_step 9 --device 1

python main-batch.py --dataset amazon2m --rand_split --metric acc --method nodeformer  --lr 1e-2 \
--weight_decay 0. --num_layers 3 --hidden_channels 64 --num_heads 1 --rb_order 1 --rb_trans identity \
--lamda 0.01 --M 50 --K 5 --use_bn --use_residual --use_gumbel --use_act --use_jk --batch_size 100000 \
--runs 5 --epochs 1000 --eval_step 9 --device 2


# image and text datasets using KNN graphs
python main.py --dataset mini --metric acc --rand_split --method nodeformer --lr 0.001\
 --weight_decay 5e-3 --num_layers 2 --hidden_channels 128 --num_heads 6\
  --rb_order 2 --rb_trans sigmoid --lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel \
   --run 5 --epochs 300 --device 2

python main.py --dataset 20news --metric acc --rand_split --method nodeformer --lr 0.001\
 --weight_decay 5e-3 --num_layers 2 --hidden_channels 64 --num_heads 4\
  --rb_order 2 --rb_trans sigmoid  --lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel \
   --run 5 --epochs 200 --device 1


# image and text datasets without input graphs
python main.py --dataset mini --metric acc --rand_split --method nodeformer --lr 0.01\
 --weight_decay 5e-3 --num_layers 2 --hidden_channels 64 --num_heads 6\
  --rb_order 0 --rb_trans sigmoid --lamda 0 --M 30 --K 10 --use_bn --use_residual --use_gumbel\
   --runs 5 --epochs 250 --device 1

python main.py --dataset 20news --metric acc --rand_split --method nodeformer --lr 0.005\
 --weight_decay 0.05 --dropout 0.3 --num_layers 2 --hidden_channels 128 --num_heads 8\
  --rb_order 0 --rb_trans sigmoid --lamda 0 --M 30 --K 10 --use_bn --use_residual --use_gumbel\
   --runs 5 --epochs 300 --device 1 