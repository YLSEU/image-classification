import torchvision.transforms as transforms
import torchvision
from randomaug import RandAugment
import torch
import numpy as np


class Cutout(object):
    """
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            # (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def get_dataset(data, size, aug, bs):
    transform_train = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:
        N = 2
        M = 14
        transform_train.transforms.insert(0, RandAugment(N, M))

    if data.lower() == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='/media/pengfei/D/YL/datasets/public/cifar10',
                                                train=True,
                                                download=True,
                                                transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
        testset = torchvision.datasets.CIFAR10(root='/media/pengfei/D/YL/datasets/public/cifar10',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8)
        num_class = 10
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif data.lower() == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='/media/pengfei/D/YL/datasets/public/cifar100',
                                                 train=True,
                                                 download=True,
                                                 transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
        testset = torchvision.datasets.CIFAR100(root='/media/pengfei/D/YL/datasets/public/cifar100',
                                                train=False,
                                                download=True,
                                                transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
        num_class = 100
    else:
        raise ValueError(f"'{data}' is not a valid dataset")

    return trainloader, testloader, num_class
