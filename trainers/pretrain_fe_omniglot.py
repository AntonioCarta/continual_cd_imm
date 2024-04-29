"""
Pretrain feature extractor for CIFAR100-superlasses experriments.
We use a slim ResNet18 pretrained on CIFAR10.
"""

import torch
import sys

from models.misc import finn_cnn

sys.path.append('.')
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from benchmarks.omniglot import AlphabetOmniglot 
from torch import nn
from avalanche.benchmarks.datasets import default_dataset_location
from torchvision.datasets import Omniglot
from avalanche.benchmarks.datasets.external_datasets.mnist import TensorMNIST



def load_mnist():
    dataset_root = default_dataset_location("mnist")
    train_set = TensorMNIST(root=str(dataset_root), train=True, download=True)
    test_set = TensorMNIST(root=str(dataset_root), train=False, download=True)
    return train_set, test_set


def load_alphabet_omniglot(image_size=28):
    transform = Compose([Resize(image_size), ToTensor(), Normalize((0.9221,), (0.2681,))])
    data = AlphabetOmniglot(train=True, transform=transform, download=True)
    eval_data = AlphabetOmniglot(train=False, transform=transform, download=True)
    return data, eval_data


def load_character_omniglot(image_size=28):
    print("WARNING: no train/test split yet")
    dataset_root = default_dataset_location("omniglot")
    transform = Compose([Resize(image_size), ToTensor(), Normalize((0.9221,), (0.2681,))])

    data = Omniglot(root=dataset_root, background=True, transform=transform, download=True)
    return data, data


@torch.no_grad()
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
    data_name = "alphabet_omniglot"
    model_name = "cnn"
    num_epochs = 200
    basedir = "/raid/carta/cl_dpmm/pretrained"

    print(f"CONFIG: data={data_name}, model={model_name}")

    if data_name == 'mnist':
        data, eval_data = load_mnist()
        num_classes = 10
        num_epochs = 10
    elif data_name == 'alphabet_omniglot':
        data, eval_data = load_alphabet_omniglot()
        num_classes = 50
    elif data_name == 'character_omniglot':
        data, eval_data = load_character_omniglot()
        num_classes = 964
    else:
        raise ValueError()

    if model_name == "ff":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(105*105, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes)
        ).cuda()
    elif model_name == "cnn":
        model = finn_cnn(num_classes).cuda()
    else:
        raise ValueError()

    gamma = 0.5
    acc_es = None
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2)

    for e in range(num_epochs):
        model.train()
        acc_es = None
        print(f"e={e}")
        for x, y in DataLoader(data, batch_size=128, shuffle=True, num_workers=8):
            opt.zero_grad()
            x, y = x.cuda(), y.cuda()
            yp = model(x)
            l = F.cross_entropy(yp, y)
            l.backward()
            opt.step()
            acc_es = (yp.argmax(dim=1) == y).float().mean().detach().item()
        
        lr_scheduler.step()
        print(f"TRAIN ACC (last)={acc_es}")
        eacc = eval(model, eval_data)
        print(f"EVAL ACC={eacc}")
        
    model.linear = torch.nn.Identity()  # remove classifier head
    os.makedirs(basedir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(basedir, f'{data_name}_{model_name}.pth'))
  