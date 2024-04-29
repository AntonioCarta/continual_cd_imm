import argparse

import setGPU

import sys

import yaml
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from benchmarks.cifar100 import fine_to_coarse
from models.cn_dpm.ndpm import Expert
from trainers.class_vae_trainer import get_autoencoder
from trainers.cn_dpm_trainer import build_cn_dpm_model
from trainers.slda_trainer import MySLDA

sys.path.append('.')

import numpy as np
import os
import torch

# -custom-written libraries
from utils.configuration import create_object_from_config, Config
from utils.misc import set_initial_seed, get_feature_extractor
from models.gen_classifier_vae import GenClassifier


def class_vae_validate(model, dataset, *, S=50):
    model.eval()
    preds = []
    for x, y, d in tqdm(DataLoader(dataset, batch_size=1, shuffle=False)):
        predicted = model.classify(x, S=S)
        preds.append(predicted)
    return torch.tensor(preds)


def preprocess_dataset(data_stream, feature_extractor, device):
    feature_extractor.eval()
    new_xs, new_ys, new_ds = [], [], []
    with torch.no_grad():
        for idx, data in enumerate(data_stream):
            # print(f"\texp {idx}")
            new_ys.extend(data.targets)
            new_ds.extend(data.domains)

            dl = DataLoader(data, batch_size=512, shuffle=False)
            for mb in dl:
                x, y = mb[0], mb[1]
                x = feature_extractor(x.to(device)).detach().cpu()
                new_xs.append(x)
    new_xs = torch.concatenate(new_xs, dim=0)
    new_ys = torch.tensor(new_ys)
    new_ds = torch.tensor(new_ds)
    return TensorDataset(new_xs, new_ys, new_ds)


def class_vae_predict(stream, config, output_dir):
    device = "cuda"

    feature_extractor = get_feature_extractor(config.feature_extractor_name)
    test_datasets = [e.dataset.eval() for e in stream]
    test_dataset = preprocess_dataset(test_datasets, feature_extractor, device)

    # ----- MODEL -----#
    class_vae = get_autoencoder(config.autoencoder_name, config)
    model = GenClassifier(class_vae, classes=config.num_classes).to(device)
    model.load_state_dict(torch.load(f"{output_dir}/model_state.pth"))
    return class_vae_validate(model, test_dataset, S=config.eval_s)


def slda_predict(stream, config, output_dir):
    device = "cuda"
    feature_extractor = get_feature_extractor(config.feature_extractor_name)
    test_datasets = [e.dataset.eval() for e in stream]
    test_dataset = preprocess_dataset(test_datasets, feature_extractor, device)

    # ----- MODEL -----#
    cl_strategy = MySLDA(
        slda_model=feature_extractor,
        criterion=CrossEntropyLoss(),
        input_size=config["strategy_config"]["params"]["input_size"],
        num_classes=config.num_classes,
        eval_mb_size=512,
        device=device,
    )
    cl_strategy.load_model(output_dir, "model_params/classifier_params")
    yp = cl_strategy.predict(test_dataset.tensors[0].cuda()).cpu()
    return torch.argmax(yp, dim=1)


def cdimm_predict(stream, config, output_dir):
    device = "cuda"
    feature_extractor = get_feature_extractor(config.feature_extractor_name)
    test_datasets = [e.dataset.eval() for e in stream]
    test_dataset = preprocess_dataset(test_datasets, feature_extractor, device)

    # ----- MODEL -----#
    model = create_object_from_config(config.classifier_config)
    model.load_state_dict(torch.load(os.path.join(output_dir, "model_params/classifier_params.pt")))
    model.eval()

    yp = []
    with torch.no_grad():
        for x, y, *_ in DataLoader(test_dataset, batch_size=512):
            yp.append(model.predict(x))
    return torch.concatenate(yp, dim=0)


def cndpm_predict(stream, config, output_dir):
    device = "cuda"
    feature_extractor = get_feature_extractor(config.feature_extractor_name)
    test_datasets = [e.dataset.eval() for e in stream]
    test_dataset = preprocess_dataset(test_datasets, feature_extractor, device)

    # ----- MODEL -----#
    model, _ = build_cn_dpm_model(config["cn_dpm_config"], output_dir)
    model_state = torch.load(os.path.join(output_dir, "model_params/classifier_params.pt"))
    # model is initialized with one expert. We need to add the others.
    num_new_experts = 0
    for pname in model_state.keys():
        if "ndpm.experts" in pname:
            eid = pname.split('.')[2]
            num_new_experts = max(num_new_experts, int(eid))
    for _ in range(num_new_experts):
        model.ndpm.experts.append(Expert(model.ndpm.config, model.ndpm.get_experts()))
        model.ndpm.prior.add_expert()

    model.load_state_dict(model_state)
    model.eval()

    yp = []
    with torch.no_grad():
        for x, y, *_ in DataLoader(test_dataset, batch_size=512):
            if x.ndim < 4:
                x = x.view(x.shape + tuple([1]*(4-x.ndim)))
            b = x.size(0)
            logits = model(x).view(b, -1)

            yp.append(logits.argmax(dim=1).cpu())

    return torch.concatenate(yp, dim=0)


def get_predictions(stream, config, output_dir):
    if "VAE" in output_dir:
        return class_vae_predict(stream, config, output_dir)
    elif "LDA" in output_dir:
        return slda_predict(stream, config, output_dir)
    elif "CD-IMM" in output_dir:
        return cdimm_predict(stream, config, output_dir)
    elif "CN-DPM" in output_dir:
        return cndpm_predict(stream, config, output_dir)
    else:
        raise ValueError("unknown model for ", output_dir)


def eval_run(args_ours, cfg_fname):
    """Function for running one continual learning experiment."""
    output_dir, _ = os.path.split(cfg_fname)

    # Set device, num_cpus and random seeds
    seed = set_initial_seed(args_ours.other_config.seed if 'seed' in args_ours.other_config else -1)
    args_ours.other_config['seed'] = seed

    # ----- DATA -----#
    for k in ['train_transform_config', 'eval_transform_config']:
        if 'params' in args_ours.scenario_config and k in args_ours.scenario_config['params']:
            transf_config = args_ours.scenario_config['params'].pop(k)
            args_ours.scenario_config['params']['_'.join(k.split('_')[:2])] = create_object_from_config(transf_config)
    scenario = create_object_from_config(args_ours.scenario_config)

    # ----- EVALUATION -----#
    yt, do = [], []
    for idx, exp in enumerate(scenario.test_stream):
        yt.extend(exp.dataset.targets)
        do.extend(exp.dataset.domains)
    yt = torch.tensor(yt)
    do = torch.tensor(do)

    yp = get_predictions(scenario.test_stream, args_ours, output_dir)
    avg_acc = (yp == yt).float().mean()
    print(f"=> Average test accuracy: {avg_acc}")
    acc_timeline = []
    for eid, exp in enumerate(scenario.train_stream):
        curr_doms = exp.dataset.domains.uniques
        acc = 0
        num_samples = 0
        for ddd in curr_doms:
            mask = (do == ddd)
            acc += (yp[mask] == yt[mask]).float().sum()
            num_samples += mask.sum()
        acc_timeline.append((acc / num_samples).item())
    print("Test ACC Timeline: ", acc_timeline)
    return avg_acc, acc_timeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_selection_dir", type=str, help="directory of model selection")
    args = parser.parse_args()
    ms_dir = args.model_selection_dir

    # config = Config.from_yaml_file(ms_dir)

    ass = []
    ats = []
    for i in range(5):
        print(f"RUN {i}")
        config = Config.from_yaml_file(f"{ms_dir}/{i}/config.yaml")
        a, at = eval_run(config, f"{ms_dir}/{i}/config.yaml")
        ass.append(a)
        ats.append(at)

    a = np.array(ass)
    print(f"AVG TEST ACC MEAN: {a.mean()}, STD: {a.std()}")
    ats = np.array(ats)
    print(f"AVG TEST ACC TIMELINE MEAN: {ats.mean(axis=0).tolist()}, STD: {ats.std(axis=0).tolist()}")


    #