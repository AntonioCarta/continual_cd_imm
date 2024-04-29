import setGPU

import sys

from avalanche.benchmarks import benchmark_with_validation_stream

sys.path.append('.')

import tqdm
import numpy as np
import os
import torch
from torch import optim
from torch.utils.data import TensorDataset

# -custom-written libraries
import models.class_vae.gido_class_vae.options_gen_classifier as options
import models.class_vae.gido_class_vae.utils as utils
import models.class_vae.gido_class_vae.define_models as define
from models.class_vae.gido_class_vae.eval import evaluate
from models.class_vae.gido_class_vae.eval import callbacks as cb
import models.class_vae.gido_class_vae.visual.plt as my_plt
from utils.configuration import Config, create_object_from_config
from utils.serialisation import to_yaml_file
from utils.misc import create_datatime_dir, set_initial_seed, get_feature_extractor
from torch.utils.data import DataLoader
from models.vae import AutoEncoder
from models.gen_classifier_vae import GenClassifier
from models.ff_vae import FFAutoEncoder


def get_autoencoder(name, args_ours):
    if name == "mnist":
        return AutoEncoder(
            image_size=args_ours.image_size, image_channels=args_ours.num_channels,
            depth=0,  # conv layers
            fc_layers=2, fc_units=85, z_dim=5
        )
    elif name == "ff_vae":
        return FFAutoEncoder(
            h_dim=args_ours.model.h_dim,
            fc_layers=args_ours.model.fc_layers,
            fc_units=args_ours.model.fc_units,
            z_dim=args_ours.model.z_dim,
            input_size=args_ours.model.input_size
        )
    else:
        raise ValueError("Unknown autoencoder architecture")


def handle_inputs():
    """Function for specifying input-options and organizing / checking them"""
    # Define input options
    parser = options.define_args(filename="main_generative", description='Train & test generative classifier.')
    parser = options.add_general_options(parser)
    parser = options.add_eval_options(parser)
    parser = options.add_task_options(parser)
    parser = options.add_model_options(parser)
    parser = options.add_train_options(parser)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options

    parser.add_argument(
        "--config-file-ours",
        type=str,
        default="configs/CIFAR100/VAE.yaml",
        help="Config yaml file",
    )
    parser.add_argument(
        "--run",
        type=int,
        help="Run of the scenario",
        default=None
    )

    args = parser.parse_args()
    options.set_defaults(args)
    options.check_for_errors(args)

    args = parser.parse_args()
    c = Config.from_yaml_file(args.config_file_ours)
    if args.run is not None:
        print(args.run)
        c.scenario_config.params['run'] = args.run

    return args, c, args.config_file_ours


def split_stream_by_class(data_stream, feature_extractor, device):
    feature_extractor.eval()
    with torch.no_grad():
        new_stream = []
        for idx, data in enumerate(data_stream):
            print(f"\texp {idx}")
            targets = data.targets
            exp_classes = np.unique(targets)
            for cls in exp_classes:
                mask = data.targets == cls
                idxs = np.arange(len(data))[mask]
                data_cls = data.subset(idxs)
                dl = DataLoader(data_cls, batch_size=512)
                ds = []
                for mb in dl:
                    x, y = mb[0], mb[1]
                    assert (y != cls).sum() == 0
                    x = feature_extractor(x.to(device)).detach().cpu()
                    ds.append(x)
                ds = torch.concatenate(ds, dim=0)
                ys = torch.zeros(ds.shape[0], dtype=torch.int32) + cls

                ds = TensorDataset(ds, ys.cpu())
                new_stream.append(ds)
    return new_stream


def train_gen_classifiers(model, train_datasets, iters=2000, epochs=None, batch_size=32,
                          feature_extractor=None, loss_cbs=list(), sample_cbs=list()):
    cuda = model._is_on_cuda()

    # Loop over all tasks.
    for train_dataset in train_datasets:
        iters_left = 1  # Initialize # iters left on data-loader(s)
        if epochs is not None:
            data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=False))
            iters = len(data_loader) * epochs
        progress = tqdm.tqdm(range(1, iters + 1))

        for batch_index in range(1, iters + 1):
            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left == 0:
                data_loader = iter(utils.get_data_loader(
                    train_dataset, batch_size, cuda=cuda, drop_last=False))
                iters_left = len(data_loader)

            x, y = next(data_loader)
            x, y = x.to('cuda'), y.to('cuda')
            assert torch.unique(y).shape[0] == 1

            if feature_extractor is not None:
                with torch.no_grad():
                    x = feature_extractor(x)

            # Select model to be trained
            model_to_be_trained = getattr(model, "vae{}".format(y[0]))

            # Train the VAE model of this class with this batch
            loss_dict = model_to_be_trained.train_a_batch(x)

            # Fire callbacks (for visualization of training-progress)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, class_id=y[0])
            for sample_cb in sample_cbs:
                if sample_cb is not None:
                    sample_cb(model_to_be_trained, batch_index, class_id=y[0])
        progress.close()


def run(args, args_ours, c_file_name):
    """Function for running one continual learning experiment."""
    if args_ours.is_grid_search:
        # after grid search, use config file directory
        output_dir, _ = os.path.split(c_file_name)
    else:
        # parse other_config
        output_dir = args_ours.other_config.output_dir
        c_file_name, _ = os.path.splitext(os.path.basename(c_file_name))
        result_path = c_file_name.replace('_', '/')
        output_dir = os.path.join(output_dir, result_path)
        output_dir = create_datatime_dir(output_dir)
        # save the config
        to_yaml_file(c.to_dict(), os.path.join(output_dir, 'config.yaml'))

    # Set device, num_cpus and random seeds
    device = "cuda"
    if args_ours.other_config.device < 0 and 'max_num_cpu' in args_ours.other_config:
        torch.set_num_threads(c.other_config.max_num_cpu)
    seed = set_initial_seed(args_ours.other_config.seed if 'seed' in args_ours.other_config else -1)
    args_ours.other_config['seed'] = seed

    # ----- DATA -----#
    for k in ['train_transform_config', 'eval_transform_config']:
        if 'params' in args_ours.scenario_config and k in args_ours.scenario_config['params']:
            transf_config = args_ours.scenario_config['params'].pop(k)
            args_ours.scenario_config['params']['_'.join(k.split('_')[:2])] = create_object_from_config(transf_config)
    scenario = create_object_from_config(args_ours.scenario_config)
    scenario = benchmark_with_validation_stream(scenario, validation_size=0.2, shuffle=True)

    train_datasets = [e.dataset.train() for e in scenario.train_stream]
    valid_datasets = [e.dataset.eval() for e in scenario.valid_stream]
    test_datasets = [e.dataset.eval() for e in scenario.test_stream]

    # ----- FEATURE EXTRACTOR -----#
    feature_extractor = get_feature_extractor(args_ours.feature_extractor_name)

    # ---- precompute features
    print("Precomputing features and splitting experiences by class...")
    train_datasets = split_stream_by_class(train_datasets, feature_extractor, device)
    valid_datasets = split_stream_by_class(valid_datasets, feature_extractor, device)
    test_datasets = split_stream_by_class(test_datasets, feature_extractor, device)
    feature_extractor = None  # we don't need it anymore

    # ----- MAIN MODEL -----#
    print("\nDefining the model...")
    class_vae = get_autoencoder(args_ours.autoencoder_name, args_ours)
    model = GenClassifier(class_vae, classes=args_ours.num_classes).to(device)

    # Separately initialize and set optimizer for each VAE
    for class_id in range(args_ours.num_classes):
        current_model = getattr(model, 'vae{}'.format(class_id))
        current_model = define.init_params(current_model, args)
        if utils.checkattr(args, "freeze_convE"):  # - freeze weights of conv-layers?
            for param in current_model.convE.parameters():
                param.requires_grad = False
            current_model.convE.frozen = True  # --> needed to ensure batchnorm-layers also do not change
        current_model.optim_list = [
            {'params': filter(lambda p: p.requires_grad, current_model.parameters()), 'lr': args.lr},
        ]
        current_model.optimizer = optim.Adam(current_model.optim_list, lr=0.001, betas=(0.9, 0.999))

    # ----- TRAINING -----#
    print("\nTraining...")
    loss_cbs = [cb._gen_classifier_loss_cb(log=args.loss_log, classes=args_ours.num_classes, visdom=None)]
    train_gen_classifiers(
        model, train_datasets, iters=args_ours.iters, epochs=1 if args.single_epochs else None,
        batch_size=args_ours.batch_size, feature_extractor=feature_extractor,
        loss_cbs=loss_cbs, sample_cbs=[None])
    torch.save(model.state_dict(), f"{output_dir}/model_state.pth")

    # ----- EVALUATION of CLASSIFIER-----#
    print("\n\nEVALUATION RESULTS:")

    def validate(*, datasets, name):
        accs = []
        for i in range(args_ours.num_classes):
            acc = evaluate.validate(
                model, datasets[i], verbose=False, allowed_classes=None, S=args_ours.eval_s,
                feature_extractor=feature_extractor,
                test_size=None if args.eval_n == 0 else args.eval_n)
            print(" - For class {}: {:.4f}".format(i + 1, acc))
            accs.append(acc)
        average_accs = sum(accs) / args_ours.num_classes
        print('=> Average accuracy over all {} classes: {:.4f}\n'.format(args_ours.num_classes, average_accs))
        # -write out to text file
        output_file = open(f"{output_dir}/acc-{name}-evalN{args.eval_n}-S{args.eval_s}.txt", 'w')
        output_file.write('{}\n'.format(average_accs))
        output_file.close()

    print("\n Accuracy of final model on valid-set:")
    validate(datasets=valid_datasets, name="valid")
    print("\n Accuracy of final model on test-set:")
    validate(datasets=test_datasets, name="test")

    # can't plot with FFVAE!
    # #----- PLOT SAMPLES of GENERATOR -----#
    # plot_name = "{}/{}.pdf".format(output_dir, 'a')
    # pp = my_plt.open_pdf(plot_name)
    # for class_id in range(args_ours.num_classes):
    #     evaluate.show_samples(model,
    #                           args_ours.num_channels, args_ours.image_size,
    #                           normalize=None, denormalize=None,
    #                           pdf=pp, visdom=None, size=100,
    #                           title="Generated samples (class_id={})".format(class_id), class_id=class_id)
    # pp.close()


if __name__ == '__main__':
    args, c, c_file_name = handle_inputs()
    run(args, c, c_file_name)
