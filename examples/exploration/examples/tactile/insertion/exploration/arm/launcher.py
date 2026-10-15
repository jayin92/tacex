# Support both `python -m ...` and direct script execution from the checkout.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[7]))

from examples.exploration.examples.utils import generate_run_commands, generate_base_command, dict_permutations
from examples.exploration.examples.tactile.insertion.exploration.arm import experiment as exp
import argparse

PROJECT_NAME = 'Jul30ArmExplorationEpisodicTraining'

entity = 'sukhijab'
_applicable_configs = {
    # 'max_steps': [1_000_000],
    'max_steps': [1_500_000],
    'feature_dim': [50],
    'hidden_dims': [1024],
    'action_repeat': [1],
    'batch_size': [256],
    'seed': list(range(3)),
    'wandb_log': [1],
    'action_cost': [0.0],
    'project_name': [PROJECT_NAME],
    'replay_buffer_size': [500_000],
    'eval_interval': [2_000],
    'init_temperature': [1.0],
    'entity_name': [entity],
    'pixels_only': [0],
    'no_rotation': [0],
    'no_gripping': [0],
    'start_grasped': [0],
    'num_init_grasp_steps': [0],
    'env_name': ['arm_exploration'],
    'mask_touch': [0],
    'save_video': [1],
    'frame_stack': [3],
    'n_steps_returns': [3],
    # 'expl_until': [0.75],
    'expl_until': [1.0],
    'switch_frequency': [-1],
    'tactile_activation': ['normal'],
    'tactile_clipping': [-1],
    'rc_loss_coefficient': [0.0],
    'multi_camera': [1],
    'grip_range': [0.1],
    # 'updates_per_step': [1],
    # 'update_freq': [1],
    # 'multi_obj': [1],
}
config_single_object = {'multi_obj': [0], 'n_objects_start': [1],
                        'n_objects_end': [1],
                        'latent_disg_weight': [0.0, 1.0], 'visual_disg_weight': [0.0, 1.0],
                        'tactile_disg_weight': [1.0]} | _applicable_configs
# config_multiple_object_curriculum = {'multi_obj': [1], 'n_objects_start': [1], 'n_objects_end': [4],
#                                     'n_objects_steps_to_end': [4_000_000],
#                                     'latent_disg_weight': [1.0], 'visual_disg_weight': [1.0],
#                                     'tactile_disg_weight': [1.0]
#                                     } | _applicable_configs

# config_multiple_object = {'multi_obj': [1], 'n_objects_start': [4], 'n_objects_end': [4],
#                          'n_objects_steps_to_end': [10]} | _applicable_configs
# _applicable_configs_less_updates = {'alg_name': ['tactile_drq_explorer', ],
#                                     'ens_lr': [3e-4],
#                                     'dyn_ent_lr': [3e-4],
#                                     'lr': [3e-4],
#                                     'expl_agent_update_period': [2],
#                                     'agent_update_period': [2],
#                                     'tau': [0.005]} | _applicable_configs

_applicable_configs_more_updates = {'alg_name': ['tactile_drq_explorer', ],
                                    # 'tactile_maxinfodrq',
                                    # 'tactile_drq'],
                                    'ens_lr': [3e-4],
                                    'dyn_ent_lr': [3e-4],
                                    'lr': [3e-4],
                                    'agent_update_period': [10_000_000],  # never update exploitation agent
                                    'tau': [0.005]}

# config_single_object = _applicable_configs_more_updates | config_single_object
# config_multiple_object_curriculum = _applicable_configs_more_updates | config_multiple_object_curriculum
config_single_object = _applicable_configs_more_updates | config_single_object
# config_multiple_object = _applicable_configs_more_updates | config_multiple_object

# disg_all_single_object = {'latent_disg_weight': [0.0, 1.0], 'visual_disg_weight': [1.0],
#                          'tactile_disg_weight': [0.0, 1.0, 5.0]} \
#                         | config_single_object

# disg_tactile_only_single_object = {'latent_disg_weight': [0.0], 'visual_disg_weight': [0.0],
#                                   'tactile_disg_weight': [1.0]} \
#                                  | config_single_object

# disg_all_multi_object = {'latent_disg_weight': [0.0], 'visual_disg_weight': [1.0],
#                         'tactile_disg_weight': [1.0]} \
#                        | config_multiple_object

# disg_all_multi_object_curriculum = {'latent_disg_weight': [0.0], 'visual_disg_weight': [1.0],
#                                    'tactile_disg_weight': [0.0, 1.0]} \
#                                   | config_multiple_object_curriculum

# per_step_update = {
#                      'update_freq': [1],
#                      'updates_per_step': [1],
#                      'training_start': [0],
#                      'expl_agent_update_period': [1],
#                      'ensemble_update_period': [1],
#                  } | config_multiple_object_curriculum

per_step_update = {
                      'update_freq': [1],
                      'updates_per_step': [1],
                      'training_start': [0],
                      'expl_agent_update_period': [1],
                      'ensemble_update_period': [1],
                  } | config_single_object

# episodic_update = {
#                     'update_freq': [250],
#                     'updates_per_step': [256],
#                     'expl_agent_update_period': [1],
#                     'ensemble_update_period': [1],
#                     'training_start': [10_000],
#                 } | config_multiple_object_curriculum

# disg_tactile_only_multi_object = {'latent_disg_weight': [0.0], 'visual_disg_weight': [0.0],
#                                  'tactile_disg_weight': [1.0]} \
#                                 | config_multiple_object
# _applicable_configs_drqv2 = {'alg_name': ['tactile_drqv2'],
#                             'ens_lr': [1e-4],
#                             'dyn_ent_lr': [1e-4],
#                             'lr': [1e-4],
#                             'sig': [-1, 0.2],
#                             'tau': [0.01]} | _applicable_configs

# _applicable_configs_maxinfodrqv2 = {'alg_name': ['tactile_maxinfodrqv2'],
#                                    'ens_lr': [1e-4],
#                                    'dyn_ent_lr': [1e-4],
#                                    'lr': [1e-4],
#                                    'tau': [0.01],
#                                    'sig': [0.2],
#                                    } | _applicable_configs

# all_flags_combinations = dict_permutations(_applicable_configs_maxinfodrqv2) \
#                        + dict_permutations(_applicable_configs_drqv2)

# all_flags_combinations = dict_permutations(disg_all_single_object) \
#                          + dict_permutations(disg_all_multi_object) \
#                          + dict_permutations(disg_tactile_only_single_object) \
#                          + dict_permutations(disg_tactile_only_multi_object)

all_flags_combinations = dict_permutations(per_step_update) # + dict_permutations(episodic_update)


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
    num_hours = 23
    generate_run_commands(command_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                          mode=args.mode, duration=f'{num_hours}:59:00', prompt=True, mem=32000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=8, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='euler', help='how to launch the experiments')

    args = parser.parse_args()
    main(args)
