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


# Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Define input options
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        type=str,
        default="results/moons/NI/dpmm/2024_03_19_12_20_03/0/config.yaml",
        help="Config yaml file",
    )

    parser.add_argument(
        "--do-plots",
        help="Make plots during the execution",
        action='store_true',
        default=False
    )

    args = parser.parse_args()
    return args


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


# Function for running one continual learning experiment
def run():
    args = handle_inputs()
    config = Config.from_yaml_file(args.config_file)
    do_plots = args.do_plots

    output_dir = config.other_config.output_dir

    # Set device, num_cpus and random seeds
    device = 'cpu'if config.other_config.device < 0 else 'cuda'
    if config.other_config.device < 0 and 'max_num_cpu' in config.other_config:
        torch.set_num_threads(config.other_config.max_num_cpu)

    seed = set_initial_seed(config.other_config.seed if 'seed' in config.other_config else -1)

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

    # ----- FEATURE EXTRACTOR -----#
    feature_extractor = get_feature_extractor(config.feature_extractor_name)

    # ---- precompute features ----#
    print("Precomputing features")
    train_datasets = precompute_feat(train_datasets, feature_extractor, device)
    val_datasets = precompute_feat(val_datasets, feature_extractor, device)
    test_datasets = precompute_feat(test_datasets, feature_extractor, device)
    feature_extractor = None  # we don't need it anymore
    device = 'cpu' # CD-IMM runs only in CPU


    # ----- MAIN MODEL -----#
    print("\nDefining the model...")
    model = create_object_from_config(config.classifier_config)
    optimiser = create_object_from_config(config.optimiser_config, params=model.parameters())

    # ----- TRAINING -----#
    print("\nTraining...")
    n_epochs = config.training_config.n_epochs
    train_CD_IMM(model, optimiser, train_datasets, n_epochs, do_plots)

    # ----- EVALUATION of CLASSIFIER-----#
    print("\n\nEVALUATION RESULTS:")
    print("\nAccuracy of final model on validation-set:")
    val_acc_dict = evaluate_CD_IMM(model, val_datasets, do_plots)
    print("\nAccuracy of final model on test-set:")
    test_acc_dict = evaluate_CD_IMM(model, test_datasets, do_plots)

    # save the model
    model_params_dir = os.path.join(output_dir, 'model_params')
    os.makedirs(model_params_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_params_dir, 'classifier_params.pt'))

    # save all the metrics
    to_json_file(val_acc_dict, os.path.join(output_dir, 'val_metrics.json'))
    to_json_file(test_acc_dict, os.path.join(output_dir, 'test_metrics.json'))

    # save benchmark
    with open(output_dir + "/benchmark.dill", 'wb') as f:
        dill.dump(scenario, f)


def train_CD_IMM(model, optimiser, train_datasets, n_epochs, do_plots):
    model.train()

    all_data_X = []
    all_data_Y = []
    for train_data in train_datasets:
        train_dataloader = DataLoader(train_data, batch_size=1000, shuffle=True)
        pbar = tqdm(range(n_epochs), desc='Epochs')

        new_task = True
        for _ in pbar:
            elbo_tot = 0

            new_epoch = True
            for x, y, *_ in train_dataloader:

                if new_task:
                    model.start_new_task(x, y)
                    new_task = False

                if new_epoch:
                    model.start_new_epoch()
                    new_epoch = False

                optimiser.zero_grad()
                elbo = model(x, y)
                elbo.backward()
                optimiser.step()
                elbo_tot += elbo.detach()

            pbar.set_postfix({'ELBO': elbo_tot.item()})

        if do_plots:
            for x, y, *_ in train_dataloader:
                all_data_X.append(x)
                all_data_Y.append(y)
            ax = plot_moons(torch.concatenate(all_data_X), torch.concatenate(all_data_Y))
            plot_model_components(model, ax)


def evaluate_CD_IMM(model, eval_dataset, do_plots):
    model.eval()
    all_data_X = []
    all_data_Y = []
    output_dict = {'task': [], 'overall': None}
    tot_samples = 0
    tot_correct_pred = 0
    with torch.no_grad():
        task_id = 0
        for test_data in eval_dataset:
            task_correct_pred = 0
            task_samples = 0
            test_dataloader = DataLoader(test_data, batch_size=512)
            for x, y, *_ in test_dataloader:
                y_pred = model.predict(x)
                task_correct_pred += torch.sum(y == y_pred)
                task_samples += y.shape[0]
                all_data_X.append(x)
                all_data_Y.append(y)

            tot_samples += task_samples
            tot_correct_pred += task_correct_pred
            task_acc = (task_correct_pred / task_samples).item()

            output_dict['task'].append({'acc': task_acc})

            task_id += 1
            print(f'Task {task_id}: acc: {100 * task_acc:.4f}')

        tot_acc = float(tot_correct_pred) / tot_samples
        output_dict['overall'] = {'acc': tot_acc}
        print(f'Overall : acc: {100 * tot_acc:.4f}')

        if do_plots:
            all_data_X = torch.concatenate(all_data_X)
            all_data_Y = torch.concatenate(all_data_Y)
            ax = plot_moons(all_data_X, all_data_Y)
            plot_model_components(model, ax)

    return output_dict


def plot_model_components(model, ax=None, idx=None):
    from utils.visualisation import plot_Gauss2D_contour
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.subplot()

    if idx is None:
        idx = list(range(model.num_classes))

    for i, m in enumerate(model.dpmm_list):
        if i in idx:
            r, mu, sigma = m.get_expected_params()
            for i in range(mu.shape[0]):
                plot_Gauss2D_contour(mu[i], sigma[i], ax_handle=ax)

    plt.show()


if __name__ == '__main__':
    run()
