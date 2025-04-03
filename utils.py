# -*- coding: utf-8 -*-

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')

    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))

    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
    _, term_width = os.popen('stty size', 'r').read().split()
except:
    term_width = 80

term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_model(args, num_class):
    from models.resnet import ResNet18, ResNet34, ResNet101, ResNet50
    from models.vgg import VGG
    from models.convmixer import ConvMixer

    print('==> Building model..')

    if args.net.lower() == 'res18':
        net = ResNet18(num_class=num_class)

    elif args.net.lower() == 'vgg':
        net = VGG('VGG19', num_class=num_class)

    elif args.net.lower() == 'res34':
        net = ResNet34(num_class=100)

    elif args.net.lower() == 'res50':
        net = ResNet50(num_class=num_class)

    elif args.net.lower() == 'res101':
        net = ResNet101(num_class=num_class)

    elif args.net.lower() == "convmixer":
        net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)

    elif args.net.lower() == "mlpmixer":
        from models.mlpmixer import MLPMixer

        net = MLPMixer(
            image_size=args.size,
            channels=3,
            patch_size=args.patch,
            dim=512,
            depth=args.mlpmixerdepth,
            num_classes=num_class
        )

    elif args.net == "vit_small":
        from models.vit_small import ViT

        net = ViT(
            image_size=args.size,
            patch_size=args.patch,
            num_classes=num_class,
            dim=int(args.dimhead),
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )

    elif args.net == "vit_tiny":
        from models.vit_small import ViT

        net = ViT(
            image_size=args.size,
            patch_size=args.patch,
            num_classes=num_class,
            dim=int(args.dimhead),
            depth=4,
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1
        )

    elif args.net == "simplevit":
        from models.simplevit import SimpleViT

        net = SimpleViT(
            image_size=args.size,
            patch_size=args.patch,
            num_classes=num_class,
            dim=int(args.dimhead),
            depth=6,
            heads=8,
            mlp_dim=512
        )

    elif args.net == "vit":
        from models.vit import ViT, ViT_Base, ViT_Huge, ViT_Large
        net = ViT_Base(img_size=args.size, num_class=num_class)

        # net = ViT(
        #     image_size=args.size,
        #     patch_size=args.patch,
        #     num_classes=num_class,
        #     dim=int(args.dimhead),
        #     depth=6,
        #     heads=8,
        #     mlp_dim=512,
        #     dropout=0.1,
        #     emb_dropout=0.1
        # )

    elif args.net == "vit_timm":
        import timm

        net = timm.create_model("vit_base_patch16_384", pretrained=True)
        net.head = nn.Linear(net.head.in_features, num_class)

    elif args.net == "cait":
        from models.cait import CaiT

        net = CaiT(
            image_size=args.size,
            patch_size=args.patch,
            num_classes=num_class,
            dim=int(args.dimhead),
            depth=6,  # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05
        )

    elif args.net == "cait_small":
        from models.cait import CaiT

        net = CaiT(
            image_size=args.size,
            patch_size=args.patch,
            num_classes=num_class,
            dim=int(args.dimhead),
            depth=6,  # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05
        )

    elif args.net == "swin":
        from models.swin import swin_t

        net = swin_t(window_size=args.patch,
                     num_classes=num_class,
                     downscaling_factors=(2, 2, 2, 1))

    elif args.net == "mobilevit":
        from models.mobilevit import mobilevit_xxs

        net = mobilevit_xxs(args.size, num_class)
    else:
        raise ValueError(f"'{args.net}' is not a valid model")

    return net
