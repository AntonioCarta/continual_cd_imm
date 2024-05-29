"""
Pretrain feature extractor for CIFAR100-superlasses experriments.
We use a slim ResNet18 pretrained on CIFAR10.
"""
import setGPU
import torch
import sys
sys.path.append('.')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torchvision
from avalanche.models import SlimResNet18
import os
from torchvision import transforms


def load_cifar100():
    _transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    _eval_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = torchvision.datasets.CIFAR100(
        root="/raid/carta/cifar100", 
        train=True, 
        transform=_transform, 
        download=True
    )
    eval_data = torchvision.datasets.CIFAR100(
        root="/raid/carta/cifar100", 
        train=False, 
        transform=_eval_transform, 
        download=True
    )
    return data, eval_data


def load_cifar10():
    _transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )
    _eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    )

    data = torchvision.datasets.CIFAR10(
        root="/raid/carta/cifar10", 
        train=True, 
        transform=_transform, 
        download=True
    )
    eval_data = torchvision.datasets.CIFAR10(
        root="/raid/carta/cifar10", 
        train=False, 
        transform=_eval_transform, 
        download=True
    )
    return data, eval_data


def eval(model, data):
    model.eval()
    acc, cnt = 0.0, 0.0
    for x, y in DataLoader(data, batch_size=512, num_workers=8):
        x, y = x.cuda(), y.cuda()
        yp = model(x)
        acc += (yp.argmax(dim=1) == y).float().sum().detach().item()
        cnt += y.shape[0]
    return acc / cnt   


if __name__ == '__main__':
    data_name = "cifar10"
    basedir = "/raid/carta/cl_dpmm/pretrained"

    if data_name == 'cifar100':
        data, eval_data = load_cifar100()
    elif data_name == 'cifar10':
        data, eval_data = load_cifar10()
    else:
        raise ValueError()

    model = SlimResNet18(nclasses=100).cuda()
    gamma = 0.5
    acc_es = None
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2)

    for e in range(200):
        model.train()
        acc_es = None
        print(f"e={e}")
        for x, y in DataLoader(data, batch_size=128, num_workers=8, shuffle=True):
            opt.zero_grad()
            x, y = x.cuda(), y.cuda()
            yp = model(x)
            l = F.cross_entropy(yp, y)
            l.backward()
            opt.step()

            ais = (yp.argmax(dim=1) == y).float().mean().detach().item()
            if acc_es is None:
                acc_es = ais
            else:
                acc_es = gamma * acc_es + (1 - gamma) * (ais)
        
        lr_scheduler.step()
        print(f"TRAIN ACC={acc_es}")
        eacc = eval(model, eval_data)
        print(f"EVAL ACC={eacc}")
        
    model.linear = torch.nn.Identity()  # remove classifier head
    os.makedirs(basedir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(basedir, f'{data_name}_resnet18.pth'))
