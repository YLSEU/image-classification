import warnings
warnings.filterwarnings('ignore')

import torch
from datas import get_dataset
from utils import get_model
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--data', default='cifar10', type=str)
    parser.add_argument('--aug', default=True, help='disable use randomaug')
    parser.add_argument('--mixup', default=True, help='add mixup augumentations')
    parser.add_argument('--net', default='res18')
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--size', default=32, type=int)

    parser.add_argument('--patch', default=4, type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default=512, type=int)
    parser.add_argument('--convkernel', default=8, type=int, help="parameter for convmixer")
    parser.add_argument('--mlpmixerdepth', default=10, type=int, help="parameter for mlpmixer")

    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader, num_class = get_dataset(data=args.data, size=args.size, aug=args.aug, bs=args.bs)
    net = get_model(args=args, num_class=num_class)
    net.to(device)
    # print(net)

    checkpoint = torch.load('./output/cifar10/res18/93.48-LR0.001_optadam_BS512_size32_patch4_epoch100/weight.t7')
    net.load_state_dict(checkpoint['model'])

    total = 0
    correct = 0

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            total += targets.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()

    print(f'Acc: {(correct / total * 100.0):.3f}, correct={correct}, total={total} ')
