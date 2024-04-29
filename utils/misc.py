import sys
import os
from pydoc import locate
import numpy as np
import torch as th
import logging
from datetime import datetime
import random
import torch
from avalanche.models import SlimResNet18

from models.misc import finn_cnn


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_seed(seed=-1):
    if seed == -1:
        seed = random.randrange(2**32-1)
    return seed


def set_initial_seed(seed):
    seed = get_seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

    return seed


def string2class(string):
    c = locate(string)
    if c is None:
        raise ModuleNotFoundError('{} cannot be found!'.format(string))
    return c


def create_datatime_dir(base_dir):
    datetime_dir = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = os.path.join(base_dir, datetime_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def get_logger(name, log_dir, file_name, write_on_console):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")

    if file_name is not None:
        # file logger
        fh = logging.FileHandler(os.path.join(log_dir, file_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if write_on_console:
        # console logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def prompt_before_overwrite(fpath):
    ans = True
    if os.path.exists(fpath):
        eprint('{} already exists! Overwrite? [y/N]'.format(fpath))
        ans = sys.stdin.readline().strip().lower()
        if ans == 'y' or ans == 'yes':
            ans = True
        elif ans == 'n' or ans == 'no':
            ans = False
        else:
            eprint('Answer not understood. The default is NO.')

    return ans


def path_exists_with_message(fpath):
    if os.path.exists(fpath):
        eprint('{} alredy exists'.format(fpath))
        return True
    else:
        return False


def get_feature_extractor(name):
    basedir = "/raid/carta/cl_dpmm"
    if name == "cifar10_resnet18":
        model_c10 = SlimResNet18(nclasses=100)
        model_c10.linear = torch.nn.Identity()
        mname = os.path.join(basedir, "pretrained/cifar10_resnet18.pth")
        model_c10.load_state_dict(torch.load(mname))
        model_c10.cuda()
        model_c10.eval()
        return model_c10
    elif name == "character_omniglot":
        model_co = finn_cnn(964)
        model_co.load_state_dict(torch.load(os.path.join(basedir, "pretrained/character_omniglot_cnn.pth")))
        model_co[-1] = torch.nn.Identity()
        model_co.cuda()
        model_co.eval()
        return model_co
    elif name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError("Unknown Feature Extractor: ", name)