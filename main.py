# https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
import torch.optim as optim
import os
import argparse
import csv
import time
from engine import train, test, save_model
from utils import progress_bar, get_model
from datas import get_dataset
import json


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--data', default='cifar10', type=str)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--opt', default="adam", type=str)
    parser.add_argument('--resume', '-r', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--aug', default=True, help='disable use randomaug')
    parser.add_argument('--mixup', default=True, help='add mixup augumentations')
    parser.add_argument('--net', default='vit')
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--patch', default=4, type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default=768, type=int)
    parser.add_argument('--convkernel', default=8, type=int, help="parameter for convmixer")
    parser.add_argument('--mlpmixerdepth', default=10, type=int, help="parameter for mlpmixer")

    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':
    args = get_args()
    save_root = f'./output/{args.data}/{args.net}/LR{args.lr}_opt{args.opt}_BS{args.bs}_size{args.size}_patch{args.patch}_epoch{args.n_epochs}'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    args_dict = vars(args)
    with open(os.path.join(save_root, "config.json"), 'w') as f:
        json.dump(args_dict, f, indent=4)

    device = 'cuda:0'

    if args.net == "vit_timm":
        size = 384
    else:
        size = args.size

    best_acc = 0
    start_epoch = 0

    # ====================== Data ==================================
    print('==> Preparing data..')
    trainloader, testloader, num_class = get_dataset(data=args.data, size=args.size, aug=args.aug, bs=args.bs)

    # ====================== model ==================================
    net = get_model(args=args, num_class=num_class)
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
  
        checkpoint = torch.load('./output/cifar10/vit/LR0.0001_optadam_BS128_size224_patch4_epoch100/weight.t7')
        net.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    if args.opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    else:
        raise ValueError(f"'{args.opt}' is not a valid optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # ======================  ==================================
    list_loss = []
    list_acc = []
    net.to(device)

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        trainloss = train(epoch=epoch,
                          trainloader=trainloader,
                          device=device,
                          criterion=criterion,
                          net=net,
                          scaler=scaler,
                          optimizer=optimizer,
                          progress_bar=progress_bar)
        val_loss, acc = test(epoch=epoch,
                             net=net,
                             device=device,
                             testloader=testloader,
                             criterion=criterion,
                             progress_bar=progress_bar,
                             optimizer=optimizer,
                             save_root=save_root)

        if acc > best_acc:
            best_acc = acc
            save_model(net=net, save_root=save_root, epoch=epoch, best_acc=best_acc)

        scheduler.step()

        list_loss.append(val_loss)
        list_acc.append(acc)

        # Write out csv
        with open(os.path.join(save_root, 'res.csv'), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss)
            writer.writerow(list_acc)
