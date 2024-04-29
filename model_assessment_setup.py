import json
import os
import argparse

import yaml

from utils.configuration import Config
from utils.misc import create_datatime_dir


def get_acc(cfg_dir):
    if "VAE" in cfg_dir:
        with open(os.path.join(cfg_dir, "acc-valid-evalN0-S10.txt"), 'r') as f:
            return float(f.readline())
    elif "LDA" in cfg_dir:
        with open(cfg_dir + f"/all_metrics.json") as f:
            d = json.load(f)
        k = "Top1_Acc_Stream/eval_phase/test_stream/Task000"
        if k in d:
            return d[k][1][-1]
        else:  # CIFAr100 doesn't have task labels...
            k = "Top1_Acc_Stream/eval_phase/test_stream"
            return d[k][1][-1]
    elif ("CD-IMM" in cfg_dir) or ("CN-DPM" in cfg_dir):
        if not os.path.exists(cfg_dir + f"/val_metrics.json"):
            return 0
        with open(cfg_dir + f"/val_metrics.json") as f:
            d = json.load(f)
        return d["overall"]["acc"]
    else:
        raise ValueError("I don't know how to extract the avg. valid accuracy")


def run():
    SEEDS = [12345, 23451, 34512, 45123, 51234]
    parser = argparse.ArgumentParser()
    parser.add_argument("model_selection_dir", type=str, help="directory of model selection")
    args = parser.parse_args()
    ms_dir = args.model_selection_dir

    best_config_fname = None
    best_valid_acc = -1
    for ddd in os.listdir(ms_dir):
        # int(ddd)
        cfg_dir = os.path.join(ms_dir, ddd)
        cfg_acc = get_acc(cfg_dir)
        if cfg_acc > best_valid_acc:
            best_valid_acc = cfg_acc
            best_config_fname = os.path.join(cfg_dir, "config.yaml")

    output_dir = os.path.split(best_config_fname)[0]  # remove filename
    output_dir = os.path.split(output_dir)[0]  # remove run id
    output_dir = os.path.split(output_dir)[0]  # remove datatime dir
    output_dir = os.path.join(output_dir, "FINAL_RUNS")
    output_dir = create_datatime_dir(output_dir)
    with open(best_config_fname, 'r') as f:
        config = yaml.safe_load(f)
    for idx in range(5):  # create and save the configs
        current_output_dir = os.path.join(output_dir, f'{idx}')
        os.makedirs(current_output_dir)
        config['other_config']['output_dir'] = current_output_dir
        config['other_config']['seed'] = SEEDS[idx]
        config['is_grid_search'] = True  # needed for correct output directory
        config['is_assessment'] = True
        with open(os.path.join(output_dir, f'{idx}', 'config.yaml'), 'w') as f:
            yaml.safe_dump(config, f)

    print(f'configurations created in {output_dir}!')
    print(f"best config: ", best_config_fname)
    print(f"best VALID acc: ", best_valid_acc)


if __name__ == '__main__':
    run()
