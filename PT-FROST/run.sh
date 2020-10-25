# example 1: run with all default hyperparameters
python frost.py

# example 2: run with class balancing, etc.
#python frost.py --n-epochs 128 --batchsize 128 --lam-x 4 --lam-u 4 --lam-c 1  --thr 0.92 --balance 4

# example 3: 
#python frost.py --n-epochs 512 --batchsize 32 --mu 7 --thr 0.97 --lam-u 4 --lr 0.04 --weight-decay 5e-4 --momentum 0.85 --balance 4 
