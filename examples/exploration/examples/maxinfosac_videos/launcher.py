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
    'ens_lr': [3e-4],
    'dyn_ent_lr': [3e-4],
    'lr': [3e-4],
}

# add other high dim tasks
configs_humanoid = {
                       'env_name': [
                           'humanoid-stand',
                           'humanoid-run',
                           'quadruped-run',
                       ],
                       'seed': [0],
                       'max_steps': [2_500_000],
                       'eval_interval': [10_000],
                       'action_repeat': [2],
                       'num_neurons': [1024],
                       'num_hidden_layers': [2],
                   } | _applicable_configs

configs_humanoid_walk = {
                            'env_name': [
                                'humanoid-walk',
                            ],
                            'seed': [4],
                            'max_steps': [2_500_000],
                            'eval_interval': [10_000],
                            'action_repeat': [2],
                            'num_neurons': [1024],
                            'num_hidden_layers': [2],
                        } | _applicable_configs

# add other low dim tasks
configs_cartpole_finger = {
                              'env_name': ['cartpole-swingup_sparse',
                                           'finger-spin',
                                           ],
                              'seed': [0],
                              'max_steps': [250_000],
                              'eval_interval': [10_000],
                              'action_repeat': [2],
                              'num_neurons': [256],
                              'num_hidden_layers': [2],
                          } | _applicable_configs

configs_others = {
                     'env_name': [
                         'hopper-hop',
                         'walker-run',
                     ],
                     'seed': [0],
                     'max_steps': [500_000],
                     'eval_interval': [10_000],
                     'action_repeat': [2],
                     'num_neurons': [256],
                     'num_hidden_layers': [2],
                 } | _applicable_configs

configs_mountaincar = {
                          'env_name': ['MountainCarContinuous-v0'],
                          'seed': [0],
                          'max_steps': [50_000],
                          'eval_interval': [1_000],
                          'action_repeat': [1],
                          'num_neurons': [256],
                          'num_hidden_layers': [2],
                      } | _applicable_configs

all_flags_combinations = dict_permutations(configs_humanoid) \
                         + dict_permutations(configs_humanoid_walk) \
                         + dict_permutations(configs_cartpole_finger) \
                         + dict_permutations(configs_others) \
                         + dict_permutations(configs_mountaincar)


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
