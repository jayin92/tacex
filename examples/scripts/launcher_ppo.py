import argparse
import copy
import os
import re
import shlex

from jaxrl2.utils.launch_util import (
    dict_permutations,
    generate_base_command,
    generate_run_commands,
    hash_dict,
)


ENTITY = 'kiten'
PROJECT_NAME = 'DSRL_pi0_Libero_May_27_23_59_Test_Touch_R7_Envs'
MODULE_NAME = 'examples.launch_train_sim'

LIBERO_ROBOT = 'Panda'
TOUCH_GRIPPER_TYPE = 'PandaGripper'
PILOT_TASK_IDS = (58,)
ALL_TASK_IDS = (58, 69, 61, 77, 55, 21, 56, 20)
PILOT_SEEDS = (0, 1, 2, 3, 4)


def _int_list_from_env(name, default):
    value = os.environ.get(name, default)
    return [int(v) for v in value.replace(',', ' ').split()]


# Reward-only DSRL-PPO. These environment and observation settings match the
# reviewer-facing LIBERO ablations, while the learner has no TacEx/MaxInfo model.
BASE_FLAGS = {
    'algorithm': 'pixel_ppo',
    'env': 'libero',
    'libero_robot': LIBERO_ROBOT,
    'dsrl_action_mode': 'noise',
    'prefix': 'dsrl_pi0_libero_ppo',
    'wandb_project': PROJECT_NAME,
    'discount': 0.999,
    'max_steps': 500000,
    'eval_interval': 20000,
    'checkpoint_interval': 20000,
    'log_interval': 1000,
    'eval_episodes': 10,
    'resize_image': 64,
    'add_tactile': 1,
    'use_touch': 1,
    'touch_gripper_type': TOUCH_GRIPPER_TYPE,
    'tactile_shape': _int_list_from_env('tactile_shape', '32 64 3'),
    'gripper_state_indices': [0, 1],
    'action_magnitude': 1.0,
    'query_freq': 20,
    'hidden_dims': [128],
    'obs_dim': 64,
    'tactile_hidden_dims': [256, 256],
    'mask_touch': 0,
    'ppo_lr': 3e-4,
    'ppo_rollout_steps': 512,
    'ppo_num_epochs': 10,
    'ppo_num_minibatches': 4,
    'ppo_clip_epsilon': 0.2,
    'ppo_gae_lambda': 0.95,
    'ppo_value_loss_coef': 0.5,
    'ppo_entropy_coef': 1e-2,
    'ppo_max_grad_norm': 0.5,
    'ppo_target_kl': 0.03,
    'ppo_init_log_std': 0.0,
    'ppo_normalize_advantages': 1,
}


OBSERVATION_PROFILES = {
    'tactile_input': {
        'add_tactile': 1,
        'use_touch': 1,
    },
    'no_tactile_input': {
        'add_tactile': 0,
        'use_touch': 0,
    },
}


# A deliberately small first-stage tuning grid. Run it on seed 0 (the --tune
# default), then confirm the chosen setting with the five pilot seeds.
PPO_TUNING_GRID = {
    'ppo_lr': [1e-4, 3e-4],
    'ppo_entropy_coef': [0.0, 1e-3, 1e-2, 3e-2],
}


def _unique(values):
    return tuple(dict.fromkeys(values))


def resolve_task_ids(task_ids=None, all_tasks=False):
    if all_tasks:
        return ALL_TASK_IDS
    if task_ids is None:
        return PILOT_TASK_IDS
    if not task_ids:
        raise ValueError('At least one LIBERO task id is required.')
    return _unique(int(task_id) for task_id in task_ids)


def resolve_seeds(seeds=None, tune=False):
    if seeds is None:
        return (0,) if tune else PILOT_SEEDS
    if not seeds:
        raise ValueError('At least one seed is required.')
    return _unique(int(seed) for seed in seeds)


def _selected_observation_profiles(observation_profile):
    if observation_profile == 'both':
        return tuple(OBSERVATION_PROFILES.items())
    try:
        return ((observation_profile,
                 OBSERVATION_PROFILES[observation_profile]),)
    except KeyError as exc:
        valid_profiles = ', '.join((*OBSERVATION_PROFILES, 'both'))
        raise ValueError(
            f'Unknown observation profile {observation_profile!r}; '
            f'choose one of: {valid_profiles}.') from exc


def build_flags(project_name=PROJECT_NAME, observation_profile='tactile_input',
                task_ids=None, all_tasks=False, seeds=None, tune=False,
                max_steps=None):
    task_ids = resolve_task_ids(task_ids, all_tasks)
    seeds = resolve_seeds(seeds, tune)

    sweep = {
        'seed': list(seeds),
        'libero_suite': ['libero_90'],
        'libero_task_id': list(task_ids),
    }
    if tune:
        sweep.update(PPO_TUNING_GRID)

    for profile_name, profile_flags in _selected_observation_profiles(
            observation_profile):
        for sweep_flags in dict_permutations(sweep):
            flags = copy.deepcopy(BASE_FLAGS)
            flags['wandb_project'] = project_name
            flags.update(sweep_flags)
            flags.update(profile_flags)
            if max_steps is not None:
                flags['max_steps'] = int(max_steps)
            suffix_flags = {
                **sweep_flags,
                'observation_profile': profile_name,
            }
            if max_steps is not None:
                suffix_flags['max_steps'] = int(max_steps)
            flags['suffix'] = hash_dict(suffix_flags)
            yield flags


def slug(value):
    value = value.replace('.', 'p').replace(',', '-')
    return re.sub(r'[^A-Za-z0-9_-]+', '-', value).strip('-')


def job_name(flags):
    seed = flags.get('seed', 'no_seed')
    task_id = flags.get('libero_task_id', 'no_task')
    suffix = flags.get('suffix', 'run')
    return slug(
        f"{flags['prefix']}_task{task_id}_{suffix}_seed{seed}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Launch reward-only DSRL-PPO LIBERO experiments.')
    parser.add_argument('--project_name', type=str, default=PROJECT_NAME)
    parser.add_argument('--entity', type=str, default=ENTITY)
    parser.add_argument('--scratch_root', type=str, default='/cluster/scratch')
    parser.add_argument('--exp_dir', type=str, default=None)
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--gpu_type', type=str, default='rtx_4090')
    parser.add_argument('--mem', type=int, default=32000)
    parser.add_argument('--duration', type=str, default=None)

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        '--task_ids', nargs='+', type=int, default=None,
        help='LIBERO task ids to run (default: 58).')
    task_group.add_argument(
        '--all_tasks', action='store_true',
        help=f'Run the full ablation task set: {ALL_TASK_IDS}.')
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=None,
        help=(
            'Seeds to run. Defaults to 0-4 for the pilot and seed 0 for '
            '--tune.'))
    parser.add_argument(
        '--observation_profile',
        choices=(*OBSERVATION_PROFILES, 'both'),
        default='tactile_input',
        help=(
            'tactile_input matches the primary ablation; no_tactile_input '
            'removes tactile sensing from learner and environment; both runs '
            'both controls.'),
    )
    parser.add_argument(
        '--tune', action='store_true',
        help=(
            'Run the first-stage PPO learning-rate x entropy-coefficient grid. '
            'Defaults to seed 0 unless --seeds is supplied.'))
    parser.add_argument(
        '--max_steps', type=int, default=None,
        help='Override the PPO environment-step budget, e.g. for a pilot.')
    parser.add_argument('--mode', type=str, default='euler',
                        choices=['euler', 'local', 'local_async'])
    parser.add_argument('--long_run', default=True, action='store_true')
    parser.add_argument('--dry', default=False, action='store_true')
    parser.add_argument('--prompt', default=True, action='store_true')
    return parser.parse_args()


def main(args):
    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = os.path.join(args.scratch_root, args.entity,
                               args.project_name)
    slurm_dir = os.path.join(exp_dir, 'slurm')

    if not args.dry:
        os.makedirs(slurm_dir, exist_ok=True)

    command_list = []
    output_file_list = []
    pre_commands = [
        f'mkdir -p {shlex.quote(exp_dir)} {shlex.quote(slurm_dir)}',
    ]
    env = {
        'EXP': exp_dir,
        'MUJOCO_GL': 'egl',
        'PYOPENGL_PLATFORM': 'egl',
        'MUJOCO_EGL_DEVICE_ID': '0',
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    }

    flags_iter = build_flags(
        project_name=args.project_name,
        observation_profile=args.observation_profile,
        task_ids=args.task_ids,
        all_tasks=args.all_tasks,
        seeds=args.seeds,
        tune=args.tune,
        max_steps=args.max_steps,
    )
    for flags in flags_iter:
        command_list.append(
            generate_base_command(
                MODULE_NAME,
                flags=flags,
                env=env,
                pre_commands=pre_commands,
            )
        )
        output_file_list.append(
            os.path.join(slurm_dir, f'{job_name(flags)}-%j.out')
        )

    duration = args.duration
    if duration is None:
        duration = '23:59:00' if args.long_run else '23:59:00'

    run_kind = 'PPO tuning' if args.tune else 'PPO pilot'
    print(f'Prepared {len(command_list)} {run_kind} {args.mode} jobs.')
    generate_run_commands(
        command_list,
        output_file_list=output_file_list,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        dry=args.dry,
        mem=args.mem,
        duration=duration,
        mode=args.mode,
        prompt=args.prompt,
        gpu_type=args.gpu_type,
    )


if __name__ == '__main__':
    main(parse_args())
