import os.path as osp
import pickle
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
import transform as T
from torchvision import transforms

from randaugment import RandomAugment
from sampler import RandomSampler, BatchSampler, SequentialSampler


def load_data_train(L=10, dspth='./dataset', seed=0, name=None):
    datalist = [
        osp.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i+1))
        for i in range(5)
    ]
    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    n_labels = L // 10
    data_x, label_x, data_u, label_u, data_all, label_all = [], [], [], [], [], []

    if seed >= 0:
        if name == None:
            name = "dataset/seeds/size"+str(L)+"seed"+str(seed)+".npy"
        print(name, "   loading")
        indices_x = np.load(name)
    
    for i in range(10):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        inds_x, inds_u, inds_all = indices[:n_labels], indices[n_labels:], indices   
        if seed >= 0:
            label_x += [labels[i] for i in inds_x]
            for j in range(n_labels):
                inds_x[j] = indices_x[i][j]
        data_x += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in inds_x
        ]
        #label_x += [labels[i] for i in inds_x]
        if seed >= 0:
            label_x += [i] * len(inds_x)
            #label_x += [i] * len(inds_x)
        else:
            label_x += [labels[j] for j in inds_x]
        data_u += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in inds_u
        ]
        label_u += [labels[i] for i in inds_u]

    data_all = [
        data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        for i in range(len(labels))
    ]
    label_all = labels

    return data_x, label_x, data_u, label_u, data_all, label_all


def load_data_val(dspth='./dataset'):
    datalist = [
        osp.join(dspth, 'cifar-10-batches-py', 'test_batch')
    ]
    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 32, 32).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels


def compute_mean_var():
    data_x, label_x, data_u, label_u, data_all, label_all = load_data_train()
    data = data_x + data_u
    data = np.concatenate([el[None, ...] for el in data], axis=0)

    mean, var = [], []
    for i in range(3):
        channel = (data[:, :, :, i].ravel() / 127.5) - 1
        #  channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        var.append(np.std(channel))

    print('mean: ', mean)
    print('var: ', var)


class Cifar10(Dataset):
    def __init__(self, data, labels, is_train=True):
        super(Cifar10, self).__init__()
        self.data, self.labels = data, labels
        self.is_train = is_train
        assert len(self.data) == len(self.labels)
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        #  mean, std = (-0.0172, -0.0356, -0.1069), (0.4940, 0.4869, 0.5231) # [-1, 1]
        if is_train:
            self.trans_reg = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
            self.trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            self.trans_strong = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
        else:
            self.trans = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        if self.is_train:
            img = Image.fromarray(im)
            return self.trans_weak(im), self.trans_strong(im), self.trans_reg(img), self.trans_reg(img), lb
        else:
            return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


def get_train_loader(batch_size, mu, mu_c, n_iters_per_epoch, L, root='dataset', seed=0, name=None):
    if name == None:
        name = "dataset/seeds/size"+str(L)+"seed"+str(seed)+".npy"
    data_x, label_x, data_u, label_u, data_all, label_all = load_data_train(L=L, dspth=root, seed=seed, name=name)
    
    ds_x = Cifar10(
        data=data_x,
        labels=label_x,
        is_train=True
    )
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)
    dl_x = torch.utils.data.DataLoader(
        ds_x,
        batch_sampler=batch_sampler_x,
        num_workers=1,
        pin_memory=True
    )
    
    ds_u = Cifar10(
        data=data_u,
        labels=label_u,
        is_train=True
    )
    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=2,
        pin_memory=True
    )
    
    ds_all = Cifar10(
        data=data_all,
        labels=label_all,
        is_train=True
    )
    #sampler_all = RandomSampler(ds_all, replacement=True, num_samples= mu_c * n_iters_per_epoch * batch_size)
    sampler_all = SequentialSampler(ds_all)
    batch_sampler_all = BatchSampler(sampler_all, batch_size * mu_c, drop_last=True)
    dl_all = torch.utils.data.DataLoader(
        ds_all,
        batch_sampler=batch_sampler_all,
        num_workers=2,
        pin_memory=True
    )
    return dl_x, dl_u, dl_all


def get_val_loader(batch_size, num_workers, pin_memory=True):
    data, labels = load_data_val()
    ds = Cifar10(
        data=data,
        labels=labels,
        is_train=False
    )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


class OneHot(object):
    def __init__(
            self,
            n_labels,
            lb_ignore=255,
        ):
        super(OneHot, self).__init__()
        self.n_labels = n_labels
        self.lb_ignore = lb_ignore

    def __call__(self, label):
        N, *S = label.size()
        size = [N, self.n_labels] + S
        lb_one_hot = torch.zeros(size)
        if label.is_cuda:
            lb_one_hot = lb_one_hot.cuda()
        ignore = label.data.cpu() == self.lb_ignore
        label[ignore] = 0
        lb_one_hot.scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(self.n_labels), *b]] = 0

        return lb_one_hot


if __name__ == "__main__":
    compute_mean_var()
    #  dl_x, dl_u = get_train_loader(64, 250, 2, 2)
    #  dl_x2 = iter(dl_x)
    #  dl_u2 = iter(dl_u)
    #  ims, lb = next(dl_u2)
    #  print(type(ims))
    #  print(len(ims))
    #  print(ims[0].size())
    #  print(len(dl_u2))
    #  for i in range(1024):
    #      try:
    #          ims_x, lbs_x = next(dl_x2)
    #          #  ims_u, lbs_u = next(dl_u2)
    #          print(i, ": ", ims_x[0].size())
    #      except StopIteration:
    #          dl_x2 = iter(dl_x)
    #          dl_u2 = iter(dl_u)
    #          ims_x, lbs_x = next(dl_x2)
    #          #  ims_u, lbs_u = next(dl_u2)
    #          print('recreate iterator')
    #          print(i, ": ", ims_x[0].size())
