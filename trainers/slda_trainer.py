from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys

from avalanche.benchmarks import benchmark_with_validation_stream

sys.path.append('.')

import torch
import argparse

from torch.nn import CrossEntropyLoss

from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training import StreamingLDA
from utils.configuration import Config, create_object_from_config
from utils.serialisation import to_json_file, to_yaml_file
from utils.misc import create_datatime_dir, set_initial_seed, get_feature_extractor
from avalanche.training.plugins import EvaluationPlugin
import dill
from avalanche.models import SlimResNet18
from torch.utils.data import DataLoader


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/CIFAR100/SLDA.yaml",
        help="Config yaml file",
    )
    args = parser.parse_args()
    return Config.from_yaml_file(args.config_file), args.config_file


class MySLDA(StreamingLDA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def make_train_dataloader(self, **kwargs):
        """we don't have task labels. Use normal dataloader"""
        self.dataloader = DataLoader(self.adapted_dataset, batch_size=self.train_mb_size)

    def _unpack_minibatch(self):
        """we don't have task labels. Removed assert shape minibatch"""
        mbatch = self.mbatch
        assert mbatch is not None
        if isinstance(mbatch, tuple):
            mbatch = list(mbatch)
            self.mbatch = mbatch

        for i in range(len(mbatch)):
            mbatch[i] = mbatch[i].to(self.device, non_blocking=True)  # type: ignore


if __name__ == "__main__":
    # set environment variables
    c, c_file_name = get_config()

    if c.is_grid_search:
        # after grid search, use config file directory
        output_dir, _ = os.path.split(c_file_name)
    else:
        # parse other_config
        output_dir = c.other_config.output_dir
        c_file_name, _ = os.path.splitext(os.path.basename(c_file_name))
        result_path = c_file_name.replace('_', '/')
        output_dir = os.path.join(output_dir, result_path)
        output_dir = create_datatime_dir(output_dir)
        # save the config
        to_yaml_file(c.to_dict(), os.path.join(output_dir, 'config.yaml'))

    device = 'cpu'

    # set the seed
    set_initial_seed(c.other_config.seed if 'seed' in c.other_config else -1)

    # build the scenario
    for k in ['train_transform_config', 'eval_transform_config']:
        if 'params' in c.scenario_config and k in c.scenario_config['params']:
            trnasf_config = c.scenario_config['params'].pop(k)
            c.scenario_config['params']['_'.join(k.split('_')[:2])] = create_object_from_config(trnasf_config)

    scenario = create_object_from_config(c.scenario_config)
    scenario = benchmark_with_validation_stream(scenario, validation_size=0.2, shuffle=True)
    metrics = [
        accuracy_metrics(minibatch=False, epoch=True, experience=True,
                         stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True)
    ]

    # build the feature extractor
    # feat_extr = create_object_from_config(c.feature_extractor_config)
    feat_extr = get_feature_extractor(c.feature_extractor_name)

    # create eval plugin
    eval_plugin = EvaluationPlugin(
        *metrics,
        loggers=[InteractiveLogger(),
                 TensorboardLogger(os.path.join(output_dir, 'tb_data'))],
    )
    # build the strategy
    csp = c['strategy_config']['params']
    cl_strategy = MySLDA(
        slda_model=feat_extr,
        criterion=CrossEntropyLoss(),
        input_size=csp["input_size"],
        num_classes=c.num_classes,
        train_mb_size=csp['train_mb_size'],
        eval_mb_size=csp['train_mb_size'],
        streaming_update_sigma=csp['streaming_update_sigma'],
        device=device,
        evaluator=eval_plugin
    )

    # We can extract the parallel train and test streams
    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # train and test loop
    results = []
    for train_task in train_stream:
        cl_strategy.train(train_task)
        results.append(cl_strategy.eval(test_stream))

    # save the model
    model_params_dir = os.path.join(output_dir, 'model_params')
    os.makedirs(model_params_dir)
    torch.save(feat_extr.state_dict(), os.path.join(model_params_dir, 'feat_extractor_params.pth'))
    cl_strategy.save_model(model_params_dir, 'classifier_params')

    # save all the metrics
    all_metrics_dict = cl_strategy.evaluator.all_metric_results
    out_file = os.path.join(output_dir, 'all_metrics.json')
    to_json_file(all_metrics_dict, out_file)
    torch.save(feat_extr.state_dict(), os.path.join(model_params_dir, 'feat_extractor_params.pth'))

    # save benchmark
    with open(output_dir + "/benchmark.dill", 'wb') as f:
        dill.dump(scenario, f)
