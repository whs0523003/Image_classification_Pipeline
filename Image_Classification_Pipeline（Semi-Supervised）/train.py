import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms

import numpy as np
import random
import sys
import argparse
import os
import time

from dataset.cifar10 import get_dataset

sys.path.append('../utils_pseudoLab/')
from TwoSampler import *
from utils_ssl import *

from ssl_networks import CNN as MT_Net
from PreResNet import PreactResNet18_WNdrop
from wideArchitectures import WRN28_2_wn

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=150, help='Training epoches')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dataset_type', default='ssl', help='How to prepare the data: only labeled data for the warmUp ("ssl_warmUp") or unlabeled and labeled for the SSL training ("ssl")')
    parser.add_argument('--train_root', default='./data', help='Root for train data')
    parser.add_argument('--labeled_samples', type=int, default=10000, help='Number of labeled samples')
    parser.add_argument('--reg1', type=float, default=0.8, help='Hyperparam for loss')
    parser.add_argument('--reg2', type=float, default=0.4, help='Hyperparam for loss')
    parser.add_argument('--download', type=bool, default=False, help='Download dataset')
    parser.add_argument('--network', type=str, default='MT_Net', help='The backbone of the network')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--seed_val', type=int, default=1, help='Seed for the validation split')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='Name of the experiment (for the output files)')
    parser.add_argument('--loss_term', type=str, default='MixUp_ep', help='The loss to use: "Reg_ep" for CE, or "MixUp_ep" for M')
    parser.add_argument('--num_classes', type=int, default=10, help='Beta parameter for the EMA in the soft labels')
    parser.add_argument('--dropout', type=float, default=0.0, help='CNN dropout')
    parser.add_argument('--load_epoch', type=int, default=0, help='Load model from the last epoch from the warmup')
    parser.add_argument('--Mixup_Alpha', type=float, default=1, help='Alpha value for the beta dist from mixup')
    parser.add_argument('--cuda_dev', type=int, default=0, help='Set to 1 to choose the second gpu')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--labeled_batch_size', default=16, type=int, metavar='N', help="Labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--validation_exp', type=str, default='False', help='Ignore the testing set during training and evaluation (it gets 5k samples from the training data to do the validation step)')
    parser.add_argument('--val_samples', type=int, default=5000, help='Number of samples to be kept for validation (from the training set))')

    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test):

    if args.validation_exp == "False":
        args.val_samples = 0

    ####################################### Train ##########################################################
    trainset, unlabeled_indexes, labeled_indexes, valset = get_dataset(args, transform_train, transform_test)

    if args.labeled_batch_size > 0 and not args.dataset_type == 'ssl_warmUp':
        print("Training with two samplers. {0} clean samples per batch".format(args.labeled_batch_size))
        batch_sampler = TwoStreamBatchSampler(unlabeled_indexes, labeled_indexes, args.batch_size, args.labeled_batch_size)
        train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    if args.validation_exp == "True":
        print("Training to choose hyperparameters --- VALIDATON MODE ---.")
        testset = valset
    else:
        print("Training to compare to the SOTA --- TESTING MODE ---.")
        testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # train and val
    print('-------> Data loading')
    print("Training with {0} labeled samples ({1} unlabeled samples)".format(len(labeled_indexes), len(unlabeled_indexes)))
    return train_loader, test_loader, unlabeled_indexes


def main(args):
    best_ac = 0.0

    #####################
    # Initializing seeds and preparing GPU
    if args.cuda_dev == 1:
        torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation
    np.random.seed(args.seed)
    #####################

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]


    transform_train = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # data lodaer
    train_loader, test_loader, unlabeled_indexes = data_config(args, transform_train, transform_test)

    if args.network == "MT_Net":
        print("Loading MT_Net...")
        model = MT_Net(num_classes = args.num_classes, dropRatio = args.dropout).to(device)

    elif args.network == "WRN28_2_wn":
        print("Loading WRN28_2...")
        model = WRN28_2_wn(num_classes = args.num_classes, dropout = args.dropout).to(device)

    elif args.network == "PreactResNet18_WNdrop":
        print("Loading preActResNet18_WNdrop...")
        model = PreactResNet18_WNdrop(drop_val = args.dropout, num_classes = args.num_classes).to(device)

    print('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    milestones = args.M

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    loss_train_epoch = []
    loss_val_epoch = []
    acc_train_per_epoch = []
    acc_val_per_epoch = []

    ####################################################################################################
    ###############################               TRAINING                ##############################
    ####################################################################################################

    for epoch in range(1, args.epoch + 1):
        st = time.time()
        scheduler.step()
        # train for one epoch
        print(args.experiment_name, args.labeled_samples)

        loss_per_epoch_train, \
        top_5_train_ac, \
        top1_train_ac, \
        train_time = train_CrossEntropy(args, model, device, \
                                        train_loader, optimizer, \
                                        epoch, unlabeled_indexes)


        loss_train_epoch += [loss_per_epoch_train]

        # test
        if args.validation_exp == "True":
            loss_per_epoch_test, acc_val_per_epoch_i = validating(args, model, device, test_loader)
        else:
            loss_per_epoch_test, acc_val_per_epoch_i = testing(args, model, device, test_loader)

        loss_val_epoch += loss_per_epoch_test
        acc_train_per_epoch += [top1_train_ac]
        acc_val_per_epoch += acc_val_per_epoch_i


if __name__ == "__main__":
    args = parse_args()
    # train
    main(args)