import random
import argparse
import numpy as np
import pandas as pd
import os
import time
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import WideResnet
from cifar import get_train_loader, get_val_loader
from label_guessor import LabelGuessor
from lr_scheduler import WarmupCosineLrScheduler
from ema import EMA
import utils

## args
parser = argparse.ArgumentParser(description=' FixMatch Training')
parser.add_argument('--wresnet-k', default=2, type=int, help='width factor of wide resnet')
parser.add_argument('--wresnet-n', default=28, type=int, help='depth of wide resnet')
parser.add_argument('--n-classes', type=int, default=10, help='number of classes in dataset')
parser.add_argument('--n-labeled', type=int, default=10, help='number of labeled samples for training')
parser.add_argument('--n-epochs', type=int, default=256, help='number of training epochs')
parser.add_argument('--batchsize', type=int, default=64, help='train batch size of labeled samples')
parser.add_argument('--mu', type=int, default=7, help='factor of train batch size of unlabeled samples')
parser.add_argument('--mu-c', type=int, default=1, help='factor of train batch size of contrastive learing samples')
parser.add_argument('--thr', type=float, default=0.95, help='pseudo label threshold')
parser.add_argument('--n-imgs-per-epoch', type=int, default=50000, help='number of training images for each epoch')
parser.add_argument('--lam-x', type=float, default=1., help='coefficient of labeled loss')
parser.add_argument('--lam-u', type=float, default=1., help='coefficient of unlabeled loss')
parser.add_argument('--lam-clr', type=float, default=1., help='coefficient of contrastive loss')
parser.add_argument('--ema-alpha', type=float, default=0.999, help='decay rate for ema module')
parser.add_argument('--lr', type=float, default=0.03, help='learning rate for training')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
parser.add_argument('--seed', type=int, default=-1, help='seed for random behaviors, no seed if negtive')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
parser.add_argument('--test', default=0, type=int, help='0 is softmax test function, 1 is similarity test function')
parser.add_argument('--bootstrap', type=int, default=8, help='Bootstrapping factor (default=8)')
parser.add_argument('--balance', type=int, default=0, help='Balance class methods to use (default=0 None)')
parser.add_argument('--delT', type=float, default=0.2, help='Class balance threshold delta (default=0.2)')
args = parser.parse_args()
print(args)

# save results
save_name_pre = '{}_E{}_B{}_LX{}_LU{}_LCLR{}_THR{}_LR{}_WD{}'.format(args.n_labeled, args.n_epochs, args.batchsize, 
    args.lam_x, args.lam_u, args.lam_clr, args.thr, args.lr, args.weight_decay)

ticks = time.time()
result_dir = 'results/' + save_name_pre + '.' + str(ticks)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def set_model():
    model = WideResnet(args.n_classes, k=args.wresnet_k, n=args.wresnet_n, feature_dim=args.feature_dim) # wresnet-28-2
    model.train()
    model.cuda()
    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss().cuda()
   
    return model, criteria_x, criteria_u
    
def train_one_epoch(
        model,
        criteria_x,
        criteria_u,
        optim,
        lr_schdlr,
        ema,
        dltrain_x,
        dltrain_u,
        dltrain_all,
        lb_guessor,
    ):
    loss_avg, loss_x_avg, loss_u_avg, loss_clr_avg = [], [], [], []
    epsilon = 0.000001
    dl_u, dl_all = iter(dltrain_u), iter(dltrain_all)
    
    for _, _, ims_all_1, ims_all_2, _ in tqdm(dl_all, desc='Training ...'):
        ims_u_weak, ims_u_strong, _, _, lbs_u = next(dl_u)

        loss_x, loss_u, loss_clr = torch.tensor(0).cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda()
        fv_1, fv_2 = torch.tensor(0).cuda(), torch.tensor(0).cuda()

        ims_u_weak = ims_u_weak.cuda()
        ims_u_strong = ims_u_strong.cuda()
        ims_all_1 = ims_all_1.cuda(non_blocking=True)
        ims_all_2 = ims_all_2.cuda(non_blocking=True)

        dl_x = iter(dltrain_x)
        ims_x_weak, _, _, _, lbs_x = next(dl_x)
        ims_x_weak = ims_x_weak.cuda()
        lbs_x = lbs_x.cuda()
        n_x, n_u, n_all = 0, 0, 0
        
        if args.lam_u >= epsilon and args.lam_clr >= epsilon:   #pseudo-labeling and Contrasive learning
            lbs_u, valid_u, mask_u  = lb_guessor(model, ims_u_weak, args.balance, args.delT)
            ims_u_strong = ims_u_strong[valid_u]
            n_x, n_u, n_all = ims_x_weak.size(0), ims_u_strong.size(0), ims_all_1.size(0)
            if n_u != 0:
                ims_x_u_all_1 = torch.cat([ims_x_weak, ims_u_strong, ims_all_1], dim=0).detach()
                ims_x_u_all_2 = torch.cat([ims_x_weak, ims_u_strong, ims_all_2], dim=0).detach()
                logits_x_u_all_1, fv_1, z_1 = model(ims_x_u_all_1)
                logits_x_u_all_2, fv_2, z_2 = model(ims_x_u_all_2)
                logits_x_u_all = (logits_x_u_all_1 + logits_x_u_all_2) / 2
                logits_x, logits_u = logits_x_u_all[:n_x], logits_x_u_all[n_x:(n_x + n_u)]
                loss_x = criteria_x(logits_x, lbs_x)
                if args.balance == 2 or args.balance == 3:
                    loss_u = (F.cross_entropy(logits_u, lbs_u, reduction='none') * mask_u).mean()
                else:
                    loss_u = criteria_u(logits_u, lbs_u)
            else:      # n_u == 0
                ims_x_all_1 = torch.cat([ims_x_weak, ims_all_1], dim=0).detach()
                ims_x_all_2 = torch.cat([ims_x_weak, ims_all_2], dim=0).detach()
                logits_x_all_1, fv_1, z_1 = model(ims_x_all_1)
                logits_x_all_2, fv_2, z_2 = model(ims_x_all_2)
                logits_x_all = (logits_x_all_1 + logits_x_all_2) / 2
                logits_x = logits_x_all[:n_x]
                loss_x = criteria_x(logits_x, lbs_x)
                loss_u = torch.tensor(0)
        elif args.lam_u >= epsilon:                             #lam_clr == 0: pseudo-labeling only
            lbs_u, valid_u, mask_u  = lb_guessor(model, ims_u_weak, args.balance, args.delT)
            ims_u_strong = ims_u_strong[valid_u]
            n_x, n_u = ims_x_weak.size(0), ims_u_strong.size(0)
            if n_u != 0:
                ims_x_u = torch.cat([ims_x_weak, ims_u_strong], dim=0).detach()
                logits_x_u, _, _ = model(ims_x_u)
                logits_x, logits_u = logits_x_u[:n_x], logits_x_u[n_x:]
                loss_x = criteria_x(logits_x, lbs_x)
                if args.balance == 2 or args.balance == 3:
                    loss_u = (F.cross_entropy(logits_u, lbs_u, reduction='none') * mask_u).mean()
                else:
                    loss_u = criteria_u(logits_u, lbs_u)
            else:     # n_u == 0
                logits_x, _, _ = model(ims_x_weak)
                loss_x = criteria_x(logits_x, lbs_x)
                loss_u = torch.tensor(0)
        else:                                                   #lam_u == 0: contrastive learning only
            n_x, n_all = ims_x_weak.size(0), ims_all_1.size(0)
            ims_x_all_1 = torch.cat([ims_x_weak, ims_all_1], dim=0).detach()
            ims_x_all_2 = torch.cat([ims_x_weak, ims_all_2], dim=0).detach()
            logits_x_all_1, fv_1, z_1 = model(ims_x_all_1)
            logits_x_all_2, fv_2, z_2 = model(ims_x_all_2)
            logits_x_all = (logits_x_all_1 + logits_x_all_2) / 2
            logits_x = logits_x_all[:n_x]
            loss_x = criteria_x(logits_x, lbs_x)
            loss_u = torch.tensor(0)
            
        if args.lam_clr >= epsilon:    
            #compute l_clr
            fv_1 = fv_1[(n_x + n_u):]
            fv_2 = fv_2[(n_x + n_u):]
            z_1 = z_1[(n_x + n_u):]
            z_2 = z_2[(n_x + n_u):]

            #[2*muc*B, D]
            z = torch.cat([z_1, z_2], dim=0)
            #[2*muc*B, 2*muc*B]
            sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / args.temperature) #denominator
            #[2*muc*B, 2*muc*B]
#            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.mu_c * args.batchsize, device=sim_matrix.device)).bool()
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.mu_c * args.batchsize, device=sim_matrix.device))
            mask = mask > 0
            #[2*muc*B, 2*muc*B - 1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * args.mu_c * args.batchsize, -1)
            #[muc*B]
            pos_sim = torch.exp(torch.sum(z_1 * z_2, dim=-1) / args.temperature) #numerator
            #[2*muc*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss_clr = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        
        #compute loss
        loss = args.lam_x * loss_x + args.lam_u * loss_u + args.lam_clr * loss_clr
        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update_params()
        lr_schdlr.step()
        
        loss_x_avg.append(loss_x.item())
        loss_u_avg.append(loss_u.item())
        loss_clr_avg.append(loss_clr.item())
        loss_avg.append(loss.item())

    ema.update_buffer()

def evaluate(ema):
    ema.apply_shadow()
    ema.model.eval()
    ema.model.cuda()

    dlval = get_val_loader(batch_size=128, num_workers=0)
    matches = []
    for ims, lbs in dlval:
        ims = ims.cuda() 
        lbs = lbs.cuda()
        with torch.no_grad():
            logits, _, _ = ema.model(ims)
            scores = torch.softmax(logits, dim=1)
            _, preds = torch.max(scores, dim=1)
            match = lbs == preds
            matches.append(match)
    matches = torch.cat(matches, dim=0).float()
    acc = torch.mean(matches)
    ema.restore()
    return acc

def test(model, memory_data_loader, test_data_loader, c, epoch):
    model.eval()
    total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, _, _ in tqdm(memory_data_loader, desc='Feature extracting'):
            logits, feature, _ = model(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            feature_labels.append(torch.tensor(torch.argmax(logits,dim=1),dtype=torch.int64))
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.cat(feature_labels, dim=0).contiguous().cpu()
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
#            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            data = data.cuda(non_blocking=True)
            _, feature, _ = model(data)
            
            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=args.k, dim=-1)
            # [B, K]
#            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices.cpu())
            sim_weight = (sim_weight / args.temperature).exp()
            
            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.k, c, device=sim_labels.device)
            # [B*K, C]

            one_hot_label = one_hot_label.scatter(-1, sim_labels.view(-1, 1), 1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.cpu().unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'
                                     .format(epoch, args.n_epochs, total_top1 / total_num * 100))
 
    return total_top1 / total_num * 100

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))

    return result_str

def sort_unlabeled(ema,numPerClass):
    ema.apply_shadow()
    ema.model.eval()
    ema.model.cuda()
    n_imgs = 2500

    _, _, dltrain_all = get_train_loader(args.n_classes, 1, n_imgs, 1, L=args.n_labeled, seed=args.seed)

    for ims_w, _, _, _, lbs in  dltrain_all:
        ims = ims_w.cuda()
        with torch.no_grad():
            logits, _, _  = ema.model(ims)
            scores = torch.softmax(logits, dim=1)
            predictions , preds = torch.max(scores, dim=1)
            top = torch.argsort(predictions, descending=True).cpu()

    preds = preds.cpu()
    predictions = predictions.cpu()
    del dltrain_all, logits

    labeledSize =args.n_classes * numPerClass
    sortByClass = np.zeros([args.n_classes, numPerClass], dtype=int)
    indx = np.zeros([args.n_classes], dtype=int)
    matches = np.zeros([args.n_classes, numPerClass], dtype=int)
    labels  = preds[top]
    samples = top

    for i in range(len(top)):
        if indx[labels[i]] < numPerClass:
            sortByClass[labels[i], indx[labels[i]]] = samples[i]
            if labels[i] == lbs[top[i]]:
                matches[labels[i], indx[labels[i]]] = 1
            indx[labels[i]] += 1
    if min(indx) < numPerClass:
        print("Counts of at least one class ", indx, " is lower than ", numPerClass)

    name = "dataset/seeds/size"+str(labeledSize)+"." + get_random_string(8) + ".npy"
    np.save(name, sortByClass[0:args.n_classes, :numPerClass])

    classAcc = 100*np.sum(matches, axis=1)/numPerClass
    print("Accuracy of the predicted pseudo-labels: top ", labeledSize,  ", ", np.mean(classAcc), classAcc )

    ema.restore()
    return name

def train():
    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize
    n_iters_all = n_iters_per_epoch * args.n_epochs #/ args.mu_c
    epsilon = 0.000001

    model, criteria_x, criteria_u = set_model()
    lb_guessor = LabelGuessor(thresh=args.thr)
    ema = EMA(model, args.ema_alpha)

    wd_params, non_wd_params = [], []
    for param in model.parameters():
        if len(param.size()) == 1:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [{'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    lr_schdlr = WarmupCosineLrScheduler(optim, max_iter=n_iters_all, warmup_iter=0)

    dltrain_x, dltrain_u, dltrain_all = get_train_loader(args.batchsize, args.mu, args.mu_c, n_iters_per_epoch, 
                                                         L=args.n_labeled, seed=args.seed)
    train_args = dict(
        model=model,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        lr_schdlr=lr_schdlr,
        ema=ema,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        dltrain_all=dltrain_all,
        lb_guessor=lb_guessor,
    )
    
    n_labeled = 1
    best_acc, top1 = -1, -1
    results = {'top 1 acc': [], 'best_acc': []}
    for e in range(args.n_epochs):
        if args.bootstrap > 1 and (e == args.n_epochs//2 or e == ((3*args.n_epochs)//4)):
            seed = 99
            n_labeled *= args.bootstrap
            name = sort_unlabeled(ema, n_labeled)
            dltrain_x, dltrain_u, dltrain_all = get_train_loader(args.batchsize, args.mu, args.mu_c, n_iters_per_epoch, 
                                                                 L=10*n_labeled, seed=seed, name=name)
            train_args = dict(
                model=model,
                criteria_x=criteria_x,
                criteria_u=criteria_u,
                optim=optim,
                lr_schdlr=lr_schdlr,
                ema=ema,
                dltrain_x=dltrain_x,
                dltrain_u=dltrain_u,
                dltrain_all=dltrain_all,
                lb_guessor=lb_guessor,
            )

        model.train()
        train_one_epoch(**train_args)
        torch.cuda.empty_cache()

        if args.test == 0 or args.lam_clr < epsilon:
            top1 = evaluate(ema) * 100
        elif args.test == 1:
            memory_data = utils.CIFAR10Pair(root='dataset', train=True, transform=utils.test_transform, download=False)
            memory_data_loader = DataLoader(memory_data, batch_size=args.batchsize, shuffle=False, num_workers=16, pin_memory=True)
            test_data = utils.CIFAR10Pair(root='dataset', train=False, transform=utils.test_transform, download=False)
            test_data_loader = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=16, pin_memory=True)
            c = len(memory_data.classes) #10
            top1 = test(model, memory_data_loader, test_data_loader, c, e)
            
        best_acc = top1 if best_acc < top1 else best_acc

        results['top 1 acc'].append('{:.4f}'.format(top1))
        results['best_acc'].append('{:.4f}'.format(best_acc))
        data_frame = pd.DataFrame(data=results)
        data_frame.to_csv(result_dir + '/' + save_name_pre + '.accuracy.csv', index_label='epoch')

        log_msg = [
            'epoch: {}'.format(e + 1),
            'top 1 acc: {:.4f}'.format(top1),
            'best_acc: {:.4f}'.format(best_acc)]
        print(', '.join(log_msg))


if __name__ == '__main__':
    train()
