# Support both `python -m ...` and direct script execution from the checkout.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from examples.exploration.examples.utils import generate_run_commands, generate_base_command, dict_permutations
from examples.exploration.examples.maxinfo_drm import experiment as exp
import argparse

PROJECT_NAME = 'MCVisualRenderTest'

entity = 'sukhijab'


_applicable_configs_dog = {
    'max_steps': [7_500_000],
    'feature_dim': [100],
    'hidden_dims': [1024],
    'action_repeat': [2],
    'batch_size': [256],
    'wandb_log': [1],
    'temp_lr': [8e-5],
    'action_cost': [0.0],
    'project_name': [PROJECT_NAME],
    'eval_interval': [10_000],
    'init_temperature': [1.0],
    'entity_name': [entity],
    'alg_name': ['maxinfodrqv2'],
    'ens_lr': [8e-5],
    'dyn_ent_lr': [8e-5],
    'lr': [8e-5],
    'tau': [0.01],
    'use_dormant_ratio': [1],
}

_applicable_configs_dog_stand = {

                              'env_name': ['dog-stand'],
                               'seed': [0],

                              } | _applicable_configs_dog

_applicable_configs_dog_walk = {

                              'env_name': ['dog-walk'],
                               'seed': [0],

                              } | _applicable_configs_dog

_applicable_configs_dog_run = {

                              'env_name': ['dog-run'],
                               'seed': [0],

                              } | _applicable_configs_dog


_applicable_configs_humanoid = {
    'max_steps': [10_000_000],
    'feature_dim': [100],
    'hidden_dims': [1024],
    'action_repeat': [2],
    'batch_size': [256],
    'wandb_log': [1],
    'temp_lr': [8e-5],
    'action_cost': [0.0],
    'project_name': [PROJECT_NAME],
    'eval_interval': [10_000],
    'init_temperature': [1.0],
    'entity_name': [entity],
    'use_dormant_ratio': [1],
    'ens_lr': [1e-4],
    'dyn_ent_lr': [1e-4],
    'lr': [8e-5],
    'tau': [0.01]
}

_applicable_configs_humanoid_stand = {

                              'env_name': ['humanoid-stand'],
                               'seed': [0],

                              } | _applicable_configs_humanoid

_applicable_configs_humanoid_walk = {

                              'env_name': ['humanoid-walk'],
                               'seed': [0],

                              } | _applicable_configs_humanoid

_applicable_configs_humanoid_run = {

                              'env_name': ['humanoid-run'],
                               'seed': [0],

                              } | _applicable_configs_humanoid


all_flags_combinations = dict_permutations(_applicable_configs_dog_stand) \
                         + dict_permutations(_applicable_configs_dog_walk) \
                         + dict_permutations(_applicable_configs_dog_run) \
                         + dict_permutations(_applicable_configs_humanoid_stand) \
                         + dict_permutations(_applicable_configs_humanoid_walk) \
                         + dict_permutations(_applicable_configs_humanoid_run)

def main(args):
    command_list = []
    logs_dir = '../'
    if args.mode == 'euler':
        logs_dir = '/cluster/scratch/'
        logs_dir += entity + '/' + PROJECT_NAME + '/'

    for flags in all_flags_combinations:
        flags['logs_dir'] = logs_dir
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)

    # submit jobs
    num_hours = 3
    generate_run_commands(command_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                          mode=args.mode, duration=f'{num_hours}:59:00', prompt=True, mem=32000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='euler', help='how to launch the experiments')

    args = parser.parse_args()
    main(args)
