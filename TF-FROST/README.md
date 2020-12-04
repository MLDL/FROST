# FROST

TensorFlow Code for FROST.


## Setup

This setup follows the setup of [FixMatch](https://github.com/google-research/fixmatch) with some additions regarding installing data.


### Install dependencies

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r requirements.txt
```

### Install datasets

Scripts for setting up the data are in the folder scripts/.  Below are the commands to set up the datasets in the folder data/.  It is hardcoded that the labeled data will be stored in the folder ./data/SSL2/.  You should cd to the TF-FROST directory before running the following commands.

```bash
mkdir data
export ML_DATA=./data
export PYTHONPATH=$PYTHONPATH:.

# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create unlabeled datasets
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
wait
cp  $ML_DATA/cifar10-train.tfrecord  $ML_DATA/cifar10p-train.tfrecord
cp  $ML_DATA/cifar10-test.tfrecord  $ML_DATA/cifar10p-test.tfrecord
cp  $ML_DATA/cifar10-train.tfrecord  $ML_DATA/cifar10imb-train.tfrecord
cp  $ML_DATA/cifar10-test.tfrecord  $ML_DATA/cifar10imb-test.tfrecord

# Create semi-supervised subsets
for seed in 0 1 2 3 4 5; do
    for size in 10 20 30 40 100 250 1000 4000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    wait
    for size in 10; do
 	CUDA_VISIBLE_DEVICES= scripts/create_imbalanced_unlabeled.py $ML_DATA/SSL2/cifar10imb $ML_DATA/cifar10imb-train.tfrecord &
	CUDA_VISIBLE_DEVICES= scripts/cifar10_prototypes.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10p $ML_DATA/cifar10p-train.tfrecord &
    done
    for size in 100 400 1000 2500 10000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
    done
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=10 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    wait
done
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
```

## Running

### Setup

All commands must be ran from the project root. The following environment variables must be defined:
```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:.
```

### Example

For example, training a FROST with 32 filters on cifar10 shuffled with `seed=3`, 10 labeled samples and 1 validation sample:
```bash
export ML_DATA="./data"
export PYTHONPATH=$PYTHONPATH:$PWD

CUDA_VISIBLE_DEVICES=0 python frost.py --filters=32 --dataset=cifar10.3@10-1 --train_dir ./experiments/frost
```

A more complete example specifying the most important input flages:

```bash
export ML_DATA="./data"
export PYTHONPATH=$PYTHONPATH:$PWD

CUDA_VISIBLE_DEVICES=0 python frost.py --train_kimg 12500 --uratio 7 --clrratio 7 --confidence 0.95 --wd 5e-4 --wu 2 --wclr 0 --batch 32 --lr 0.03 --temperature 0.5 --arch resnet --filters 32 --scales 3 --repeat 4 --dataset=cifar10.3@10-1 --train_dir experiments/ROSS/frostcifar10.3@10-1resnetIter12550U7CL7C0.95WD5e-4WU2Wclr0BS32LR0.03M0.9T0.5Ad.d.dB0D0BF16SCH1CAug0_0 --augment d.d.d --mom 0.9 --boot_factor 16 --balance 0 --delT 0 --boot_schedule 1 --clrDataAug 0 
```

The various input flags are described in the bottom of the file frost.py.  
The parameter train_kimg is the training duration in kibi-samples (as described in libml/train.py).
Available labelled sizes are 10, 20, 30, 40, 100, 250, 1000, 4000.
For validation, available sizes are 1, 5000.
Possible shuffling seeds are 1, 2, 3, 4, 5 and 0 for no shuffling (0 is not used in practiced since data requires to be
shuffled for gradient descent to work properly).
All training runs with FROST were performed on a single GPU.


#### Flags

The `--augment` flag can use a little more explanation. It is composed of 3 values, for example `d.d.d`
(`d`=default augmentation, for example shift/mirror, `x`=identity, e.g. no augmentation, `ra`=rand-augment,
 `rac`=rand-augment + cutout):
- the first `d` refers to data augmentation to apply to the labeled example. 
- the second `d` refers to data augmentation to apply to the weakly augmented unlabeled example. 
- the third `d` refers to data augmentation to apply to the strongly augmented unlabeled example. For the strong
augmentation, `d` is followed by `CTAugment` for `frost.py` and code inside `cta/`, `cta_boss/`, `cta_frost/` folders.



### Valid dataset names
```bash
for dataset in cifar10 cifar10imb svhn svhn_noextra; do
for seed in 0 1 2 3 4 5; do
for valid in 1 5000; do
for size in 10 20 30 40 100 250 1000 4000; do
    echo "${dataset}.${seed}@${size}-${valid}"
done; done; done; done

for dataset in cifar10p cifar10imb; do
for seed in 0 1 2 3 4 5; do
for valid in 1 5000; do
for size in 10; do
    echo "${dataset}.${seed}@${size}-${valid}"
done; done; done; done

for seed in 1 2 3 4 5; do
for valid in 1 5000; do
    echo "cifar100.${seed}@100-${valid}"
    echo "cifar100.${seed}@10000-${valid}"
done; done

for seed in 1 2 3 4 5; do
for valid in 1 5000; do
    echo "stl10.${seed}@10-${valid}"
    echo "stl10.${seed}@1000-${valid}"
done; done
echo "stl10.1@5000-1"
```

Cifar10p corresponds to the Cifar10 dataset with handpicked prototypes for the labeled data.  Currently set up for size 10.
Cifar10imb creates an imbalanced unlabeled dataset, where the number of training examples for classes 1 to 10 are: 5000, 5000, 2487, 1651, 1228, 980, 810, 694, 606, 532, respectively. 

## Running other algorithms

This repository contains code for running BOSS, FixMatch, ReMixMatch, MixMatch, Mean Teachers, UDA, VAT, and the Pi Model.  In addition, the folder fully_supervised/ contains the code for supervised training and the subfolder runs/ contains the shell script all.sh with the command lines.

This repository contains folders cta/, cta_boss/, and cta_frost/.  These codes are algorithm specific and are used by FixMatch, BOSS, and FROST, respectively.


## Monitoring training progress

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training
process:

```bash
tensorboard.sh --port 6007 --logdir ./experiments
```

## Adding datasets
You can add custom datasets into the codebase by taking the following steps:

1. Add a function to acquire the dataset to `scripts/create_datasets.py` similar to the present ones, e.g. `_load_cifar10`. 
You need to call `_encode_png` to create encoded strings from the original images.
The created function should return a dictionary of the format 
`{'train' : {'images': <encoded 4D NHWC>, 'labels': <1D int array>},
'test' : {'images': <encoded 4D NHWC>, 'labels': <1D int array>}}` .
2. Add the dataset to the variable `CONFIGS` in `scripts/create_datasets.py` with the previous function as loader. 
You can now run the `create_datasets` script to obtain a tf record for it.
3. Use the `create_unlabeled` and `create_split` script to create unlabeled and differently split tf records as above in the *Install Datasets* section.
4. In `libml/data.py` add your dataset in the `create_datasets` function. The specified "label" for the dataset has to match
the created splits for your dataset. You will need to specify the corresponding variables if your dataset 
has a different # of classes than 10 and different resolution and # of channels than 32x32x3
5. In `libml/augment.py` add your dataset to the `DEFAULT_AUGMENT` variable. Primitives "s", "m", "ms" represent mirror, shift and mirror+shift. 
