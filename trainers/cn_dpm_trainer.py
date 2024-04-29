import sys
# sys.path.append("/home/carta/avalanche_dev")
# print(sys.path)

sys.path.append('.')

#import setGPU
import tqdm
import os
import torch
from torch import optim
import argparse
from torch.utils.data import TensorDataset
from tqdm import tqdm
# -custom-written libraries
from utils.configuration import Config, create_object_from_config
from utils.serialisation import to_yaml_file, to_json_file
from utils.misc import create_datatime_dir, set_initial_seed, get_feature_extractor
from torch.utils.data import DataLoader
from benchmarks.moons import plot_moons
import dill
from avalanche.benchmarks import benchmark_with_validation_stream
import yaml
from tensorboardX import SummaryWriter
from models.cn_dpm import MODEL


# Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Define input options
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        type=str,
        default="results/moons/CN-DPM/ni/2024_03_26_14_25_01/0/config.yaml",
        help="Config for the experiment",
    )

    args = parser.parse_args()

    return args


def build_cn_dpm_model(config, log_dir):

    # Set log directory
    config['log_dir'] = log_dir
    #if os.path.exists(log_dir):
    #    print(f'WARNING: {log_dir} already exists')
    #    input('Press enter to continue')

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    config_save_path = os.path.join(config['log_dir'], 'config.yaml')
    episode_save_path = os.path.join(config['log_dir'], 'episode.yaml')
    #yaml.dump(config, open(config_save_path, 'w'))
    #yaml.dump(episode, open(episode_save_path, 'w'))
    #print('Config & episode saved to {}'.format(config['log_dir']))

    # Build components
    #data_scheduler = DataScheduler(config)
    writer = SummaryWriter(config['log_dir'])
    model = MODEL[config['model_name']](config, writer)
    model.to(config['device'])
    return model, writer


def train_cn_dpm(model, train_datasets, n_epochs, batch_size, writer):
    model.train()
    step = 0
    for train_data in train_datasets:
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(n_epochs), desc='Epochs'):

            for x, y, *_ in train_dataloader:
                if x.ndim < 4:
                    x = x.view(x.shape + tuple([1]*(4-x.ndim)))

                step += 1
                model.learn(x, y, t=None, step=step)

                # Evaluate experts of  the model's DPMoE
                n_experts = len(model.ndpm.experts) - 1
                writer.add_scalar('num_experts', n_experts, step)


@torch.no_grad()
def precompute_feat(data_stream, feature_extractor, device):
    feature_extractor = feature_extractor.to(device)
    new_stream = []
    for data in data_stream:
        dl = DataLoader(data, batch_size=512)
        new_feat = []
        new_y = []
        for x, y, *_ in dl:
            feat = feature_extractor(x.to(device)).detach().cpu()
            new_feat.append(feat)
            new_y.append(y)

        new_feat = torch.concatenate(new_feat)
        new_y = torch.concatenate(new_y)
        new_stream.append(TensorDataset(new_feat, new_y.cpu()))
    return new_stream


def evaluate_cn_dpm(model, eval_dataset, eval_batch_size):
    model.eval()

    totals = []
    corrects_1 = []
    output_dict = {'task': [], 'overall': None}
    with torch.no_grad():
        task_id = 0
        for test_data in eval_dataset:
            correct_1 = 0
            total = 0
            test_dataloader = DataLoader(test_data, batch_size=eval_batch_size)
            for x, y, *_ in test_dataloader:
                if x.ndim < 4:
                    x = x.view(x.shape + tuple([1]*(4-x.ndim)))

                b = x.size(0)
                with torch.no_grad():
                    logits = model(x).view(b, -1)
                # [B, K]
                #_, pred_topk = logits.topk(K, dim=1)
                #correct_topk = (
                #        pred_topk.cpu() == y.view(b, -1).expand_as(pred_topk)
                #).float()
                pred = torch.argmax(logits, -1)
                correct_1 += (pred.view(-1).cpu()==y.cpu()).sum()
                #correct_k += correct_topk[:, :K].view(-1).cpu().sum()
                total += x.size(0)
            totals.append(total)
            corrects_1.append(correct_1)
            accuracy_1 = (correct_1 / total).item()

            task_id += 1
            output_dict['task'].append({'acc': accuracy_1})
            print(f'Task {task_id}: acc: {100 * accuracy_1:.4f}')

        # Overall accuracy
        total = sum(totals)
        correct_1 = sum(corrects_1)
        #correct_k = sum(corrects_k)
        accuracy_1 = (correct_1 / total).item()
        #accuracy_k = correct_k / total
        output_dict['overall'] = {'acc': accuracy_1}#, f'{K}-acc': accuracy_k}

        print(f'Overall: acc: {100 * accuracy_1:.4f}')# \t|\t {K}-acc: {100 * accuracy_k:.4f}')

    return output_dict


# Function for running one continual learning experiment
def run():
    args = handle_inputs()

    config = Config.from_yaml_file(args.config_file)
    cn_dpm_config = config.cn_dpm_config

    output_dir = config.other_config.output_dir

    # Set device, num_cpus and random seeds
    device = 'cpu'if config.other_config.device < 0 else 'cuda'
    if config.other_config.device < 0 and 'max_num_cpu' in config.other_config:
        torch.set_num_threads(config.other_config.max_num_cpu)

    seed = set_initial_seed(config.other_config.seed if 'seed' in config.other_config else -1)
    cn_dpm_config['device'] = device

    # ----- DATA -----#
    for k in ['train_transform_config', 'eval_transform_config']:
        if 'params' in config.scenario_config and k in config.scenario_config['params']:
            transf_config = config.scenario_config['params'].pop(k)
            config.scenario_config['params']['_'.join(k.split('_')[:2])] = create_object_from_config(transf_config)
    scenario = create_object_from_config(config.scenario_config)
    scenario = benchmark_with_validation_stream(scenario, validation_size=0.2, shuffle=True)
    train_datasets = [e.dataset.train() for e in scenario.train_stream]
    val_datasets = [e.dataset.eval() for e in scenario.valid_stream]
    test_datasets = [e.dataset.eval() for e in scenario.test_stream]

    #----- FEATURE EXTRACTOR -----#
    feature_extractor = get_feature_extractor(config.feature_extractor_name)
    # ---- precompute features ----#
    print("Precomputing features")
    train_datasets = precompute_feat(train_datasets, feature_extractor, device)
    val_datasets = precompute_feat(val_datasets, feature_extractor, device)
    test_datasets = precompute_feat(test_datasets, feature_extractor, device)
    feature_extractor = None  # we don't need it anymore

    # ----- MAIN MODEL -----#
    print("\nDefining the model...")
    model, writer = build_cn_dpm_model(cn_dpm_config, output_dir)

    # ----- TRAINING -----#
    print("\nTraining...")
    n_epochs = config.training_config.n_epochs
    train_batch_size = cn_dpm_config['batch_size']
    train_cn_dpm(model, train_datasets, n_epochs, train_batch_size, writer)

    # ----- EVALUATION of CLASSIFIER-----#
    eval_batch_size = cn_dpm_config['eval_batch_size']
    print("\n\nEVALUATION RESULTS:")
    print("\nAccuracy of final model on validation-set:")
    val_acc_dict = evaluate_cn_dpm(model, val_datasets, eval_batch_size)
    print("\nAccuracy of final model on test-set:")
    test_acc_dict = evaluate_cn_dpm(model, test_datasets, eval_batch_size)

    # save the model
    model_params_dir = os.path.join(output_dir, 'model_params')
    os.makedirs(model_params_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_params_dir, 'classifier_params.pt'))

    # save all the metrics
    # save all the metrics
    to_json_file(val_acc_dict, os.path.join(output_dir, 'val_metrics.json'))
    to_json_file(test_acc_dict, os.path.join(output_dir, 'test_metrics.json'))

    # save benchmark
    with open(output_dir + "/benchmark.dill", 'wb') as f:
        dill.dump(scenario, f)


if __name__ == '__main__':
    run()
