import os
import argparse

from utils.configuration import Config
from utils.misc import create_datatime_dir, get_seed


def handle_inputs():
    # Define input options
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file",
        type=str,
        help="Config yaml file to build the model selection grid",
    )

    args = parser.parse_args()
    c_file_path = args.config_file
    c = Config.from_yaml_file(c_file_path)

    return args, c, c_file_path


def run():
    args, config, c_file_path = handle_inputs()
    output_dir = config.other_config.output_dir
    c_file_name, _ = os.path.splitext(os.path.basename(c_file_path))
    result_path = c_file_name.replace('_', '/')
    output_dir = os.path.join(output_dir, result_path)
    output_dir = create_datatime_dir(output_dir)

    # build the grid
    config_list = config.build_config_grid()
    n_configs = len(config_list)
    for i, c in enumerate(config_list):
        # save the configs
        current_output_dir = os.path.join(output_dir, f'{i}')
        os.makedirs(current_output_dir)
        c.other_config['output_dir'] = current_output_dir
        c.other_config['seed'] = get_seed(c.other_config.seed)
        c.is_grid_search = True
        c.to_yaml_file(os.path.join(output_dir, f'{i}', 'config.yaml'))

    print(f'{n_configs} configurations created!')
    print(f"directory: {output_dir}")


if __name__ == '__main__':
    run()
