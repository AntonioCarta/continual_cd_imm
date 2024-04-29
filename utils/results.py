import os
import os.path as osp
import numpy as np
from .configuration import Config
from .serialisation import from_json_file, from_torch_file
from functools import reduce


class ModelSelectionResult:

    def __init__(self, exp_dir_path):

        self.exp_dir_path = exp_dir_path
        self.exp_config = Config.from_yaml_file(os.path.join(exp_dir_path, 'grid.yaml'))
        self.grid = self.exp_config.get_grid()
        self.grid_shape = tuple([len(v) for k, v in self.grid.items()])
        self.count_exp_finished = 0
        self.tot_exp = 0
        self.dict_results = {}
        self.update()

    def update(self):
        self.count_exp_finished = 0

        # compute the number of config dir
        list_config_dir_name = [x for x in os.listdir(self.exp_dir_path) if x.startswith('config')]
        n_configs = len(list_config_dir_name)
        self.n_configs = n_configs

        # compute the number of split dir
        list_split_dir_name = [x for x in os.listdir(osp.join(self.exp_dir_path, 'config_0')) if x.startswith('split')]
        n_splits = len(list_split_dir_name)
        self.n_splits = n_splits

        self.dict_results = {}
        self.tot_exp = n_configs * n_splits

        for config_dir_name in list_config_dir_name:
            config_idx = int(config_dir_name.split('_')[-1])
            for split_dir_name in list_split_dir_name:
                split_idx = int(split_dir_name.split('_')[-1])
                split_dir_path = osp.join(self.exp_dir_path, config_dir_name, split_dir_name)

                results_file = osp.join(split_dir_path, 'training_results.json')
                if osp.exists(results_file):
                    self.count_exp_finished += 1
                    current_res_d = from_json_file(results_file)
                    for k in current_res_d:
                        if k not in self.dict_results:
                            self.dict_results[k] = np.full((n_configs, n_splits), fill_value=np.nan)

                        self.dict_results[k][config_idx, split_idx] = current_res_d[k]

    # TODO: implement methods to ravel/unravel
    def get_config_detail(self, config_idx):
        idx_list = np.unravel_index(config_idx, self.grid_shape)
        return {k: v[idx_list[i]] for i, (k,v) in enumerate(self.grid.items())}

    def __getitem__(self, key_list):
        metric, config_idx, split_idx = tuple(key_list)
        if metric in self.dict_results:
            return self.dict_results[metric][config_idx, split_idx]
        else:
            # it is a file to load
            file_path = osp.join(self.exp_dir_path, f'config_{config_idx}', f'split_{split_idx}', f'{metric}.pt')
            return from_torch_file(file_path)

    def get_test_results(self):
        if 'validation_metric' not in self.dict_results:
            return None, None
        best_config_idx = np.nanargmax(self.dict_results['validation_metric'], axis=0)
        return self.dict_results['test_metric'][best_config_idx, np.arange(self.n_splits)], best_config_idx

    def get_WRONG_test_results(self):
        if 'validation_metric' not in self.dict_results:
            return None, None
        best_config_idx = np.nanargmax(np.nanmean(self.dict_results['validation_metric'], axis=1), axis=0)
        return self.dict_results['test_metric'][best_config_idx, :], best_config_idx

    def get_best_idx(self):
        best_config_idx = np.nanargmax(np.nanmean(self.dict_results['validation_metric'], axis=1), axis=0)
        return best_config_idx

    def get_avg_val_results(self, k_list):
        if 'validation_metric' not in self.dict_results:
            return None, None

        avg_res = self.dict_results['validation_metric'].mean(-1).reshape(self.grid_shape)
        labels = []
        for i, k in enumerate(self.grid):
            if k not in k_list:
                avg_res = np.nanmax(avg_res, i, keepdims=True)
            else:
                labels.append(self.grid[k])

        return avg_res.squeeze(), labels

    def __str__(self):
        self.update()
        s = ''
        s += '-' * 50 + '\n'
        s += f"Config {self.count_exp_finished}/{self.tot_exp}\n"
        test_res, best_config_idx = self.get_WRONG_test_results()
        test_res *= 100
        s += f"Test result: {np.nanmean(test_res):0.2f} ± {np.nanstd(test_res):0.2f}\n"
        s += f"Best config idx: {best_config_idx}\n"
        s += '-' * 50 + '\n'
        return s


