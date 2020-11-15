
# FROST

PyTorch Code for the paper: "FROST: Faster and more Robust One-shot Semi-supervised Training".


## Dataset
### Before training, download cifar-10 dataset: 

    cd dataset
    wget -c http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
    rm -f cifar-10-python.tar.gz
    
## Dependencies
### python 3 is required to run the script. Dependencies include torch, torchvision, pandas, tqdm, cv2. 

### For AWS AMI: if you use AWS AMI, we suggest using anaconda. All you need to do is to activate pytorch 3 and all the required packages should be there already.
    source activate pytorch_p36

## Train & Test the model
### parameters can be set in the frost.py flags

    # example 1: run with all default parameters
    python frost.py

    # example 2: run with class balancing, etc.
    #python frost.py --n-epochs 512 --batchsize 128 --lam-x 4 --lam-u 4 --lam-c 1  --thr 0.92 --balance 4

    # example 3: 
    #python frost.py --n-epochs 256 --batchsize 256 --mu 7 --thr 0.97 --lam-u 4 --lr 0.04 --weight-decay 5e-4 --momentum 0.85 --balance 4 