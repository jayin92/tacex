# Support both `python -m ...` and direct script execution from the checkout.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from examples.exploration.examples.utils import generate_run_commands, generate_base_command, dict_permutations
from examples.exploration.examples.maxinfosac_videos import experiment as exp
import argparse

PROJECT_NAME = 'MBPOSACVideos'

entity = 'sukhijab'
_applicable_configs = {
    'batch_size': [256],
    'wandb_log': [1],
    'action_cost': [0.0],
    'project_name': [PROJECT_NAME],
    'entity_name': [entity],
    'alg_name': ['maxinfosac'],
    'ens_lr': [5e-5],  # 5e-5
    'dyn_ent_lr': [5e-5],  # 5e-5
    'lr': [5e-5],  # 5e-5
    'use_bronet': [0],
    'init_temperature_dyn_entropy': [2.0],
    'replay_buffer_size': [2_500_000],
}

# add other high dim tasks
configs_humanoid = {
                       'env_name': [
                           'humanoid_bench/h1-stand-v0',
                           'humanoid_bench/h1-run-v0',
                           'humanoid_bench/h1-walk-v0',
                       ],
                       'seed': [4],
                       'max_steps': [5_000_000],
                       'eval_interval': [10_000],
                       'action_repeat': [1],
                       'num_neurons': [512],
                       'num_hidden_layers': [2],
                   } | _applicable_configs

all_flags_combinations = dict_permutations(configs_humanoid)


def main(args):
    command_list = []
    logs_dir = '../'
    if args.mode == 'euler':
        logs_dir = '/global/scratch/users/'
        logs_dir += 'csferrazza' + '/' + PROJECT_NAME + '/'

    for flags in all_flags_combinations:
        flags['logs_dir'] = logs_dir
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    num_hours = 23
    generate_run_commands(command_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                          mode=args.mode, duration=f'{num_hours}:59:00', prompt=True, mem=32000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='euler', help='how to launch the experiments')

    args = parser.parse_args()
    main(args)
