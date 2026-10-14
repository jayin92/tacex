"""Launch the two summed-taxel rebuttal baselines on seven LIBERO tasks.

This launcher is intentionally separate from every existing experiment grid.
It creates five seeds per task for:

* MaxInfoSAC uncertainty over one summed-taxel signal.
* Pixel SAC trained only on that summed-taxel signal as reward.
"""

import argparse
import copy
import os
import re
import shlex

from jaxrl2.utils.launch_util import (
    generate_base_command,
    generate_run_commands,
    hash_dict,
)


ENTITY = "kiten"
PROJECT_NAME = "DSRL_pi0_Libero_Taxel_Baselines_7Tasks"
MODULE_NAME = "examples.launch_train_sim"

LIBERO_ROBOT = "Panda"
TOUCH_GRIPPER_TYPE = "PandaGripper"
DEFAULT_TASK_IDS = (69, 61, 77, 55, 21, 56, 20)
DEFAULT_SEEDS = tuple(range(5))
DEFAULT_BASELINES = ("maxinfo_taxel_sum", "sac_taxel_reward")


def _int_list_from_env(name, default):
    value = os.environ.get(name, default)
    return [int(item) for item in value.replace(",", " ").split()]


COMMON_FLAGS = {
    "env": "libero",
    "libero_robot": LIBERO_ROBOT,
    "dsrl_action_mode": "noise",
    "batch_size": 256,
    "discount": 0.999,
    "max_steps": 500000,
    "checkpoint_interval": 20000,
    "eval_interval": 20000,
    "log_interval": 1000,
    "eval_episodes": 10,
    "multi_grad_step": 20,
    "start_online_updates": 500,
    "resize_image": 64,
    "add_tactile": 1,
    "use_touch": 1,
    "touch_gripper_type": TOUCH_GRIPPER_TYPE,
    "tactile_shape": _int_list_from_env("tactile_shape", "32 64 3"),
    "gripper_state_indices": [0, 1],
    "action_magnitude": 1.0,
    "query_freq": 20,
    "hidden_dims": [128],
    "tactile_hidden_dims": [256, 256],
    "mask_touch": 0,
}


BASELINE_FLAGS = {
    "maxinfo_taxel_sum": {
        "algorithm": "pixel_maxinfosac_taxel_sum",
        "prefix": "dsrl_pi0_libero_maxinfo_taxel_sum",
        "dyn_ent_lr": 3e-4,
        "init_dyn_ent_temperature": 1.0,
        "model_lr": 1e-3,
        "model_wd": 1e-4,
        "model_hidden_dims": [256, 256],
        "num_model_heads": 5,
        "model_noise_var": 1.0,
        "predict_reward": 1,
        "predict_diff": 1,
        "backup_entropy": 1,
        "model_obs_key": "state",
        "obs_dim": 64,
        "ensemble_disagreement_modalities": "tactile",
        "mask_expl_critic": 1,
    },
    "sac_taxel_reward": {
        "algorithm": "pixel_sac_taxel_reward",
        "prefix": "dsrl_pi0_libero_sac_taxel_reward",
    },
}


def build_flags(
        project_name=PROJECT_NAME,
        baselines=DEFAULT_BASELINES,
        task_ids=DEFAULT_TASK_IDS,
        seeds=DEFAULT_SEEDS,
        max_steps=None,
):
    unknown = sorted(set(baselines) - set(BASELINE_FLAGS))
    if unknown:
        raise ValueError(
            f"Unknown baselines {unknown}; choose from {tuple(BASELINE_FLAGS)}.")

    for baseline in baselines:
        for task_id in task_ids:
            for seed in seeds:
                flags = copy.deepcopy(COMMON_FLAGS)
                flags.update(BASELINE_FLAGS[baseline])
                flags.update({
                    "wandb_project": project_name,
                    "libero_suite": "libero_90",
                    "libero_task_id": int(task_id),
                    "seed": int(seed),
                })
                if max_steps is not None:
                    flags["max_steps"] = int(max_steps)
                flags["suffix"] = hash_dict({
                    "baseline": baseline,
                    "task_id": int(task_id),
                    "seed": int(seed),
                })
                yield flags


def slug(value):
    value = value.replace(".", "p").replace(",", "-")
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")


def job_name(flags):
    return slug(
        f"{flags['prefix']}_task{flags['libero_task_id']}_"
        f"seed{flags['seed']}_{flags['suffix']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch isolated summed-taxel DSRL rebuttal baselines.")
    parser.add_argument("--project_name", default=PROJECT_NAME)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--baselines", nargs="+", choices=tuple(BASELINE_FLAGS),
                        default=list(DEFAULT_BASELINES))
    parser.add_argument("--task_ids", nargs="+", type=int,
                        default=list(DEFAULT_TASK_IDS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--scratch_root", default="/cluster/scratch")
    parser.add_argument("--exp_dir", default=None)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu_type", default="rtx_4090")
    parser.add_argument("--mem", type=int, default=32000)
    parser.add_argument("--duration", default="23:59:00")
    parser.add_argument("--mode", choices=("euler", "local", "local_async"),
                        default="euler")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--prompt", default=True, action="store_true")
    return parser.parse_args()


def main(args):
    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = os.path.join(args.scratch_root, args.entity, args.project_name)
    slurm_dir = os.path.join(exp_dir, "slurm")

    if not args.dry:
        os.makedirs(slurm_dir, exist_ok=True)

    pre_commands = [
        f"mkdir -p {shlex.quote(exp_dir)} {shlex.quote(slurm_dir)}",
    ]
    env = {
        "EXP": exp_dir,
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    flags_list = list(build_flags(
        project_name=args.project_name,
        baselines=args.baselines,
        task_ids=args.task_ids,
        seeds=args.seeds,
        max_steps=args.max_steps,
    ))
    command_list = [
        generate_base_command(
            MODULE_NAME,
            flags=flags,
            env=env,
            pre_commands=pre_commands,
        )
        for flags in flags_list
    ]
    output_file_list = [
        os.path.join(slurm_dir, f"{job_name(flags)}-%j.out")
        for flags in flags_list
    ]

    print(f"Prepared {len(command_list)} summed-taxel {args.mode} jobs.")
    generate_run_commands(
        command_list,
        output_file_list=output_file_list,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        dry=args.dry,
        mem=args.mem,
        duration=args.duration,
        mode=args.mode,
        prompt=args.prompt,
        gpu_type=args.gpu_type,
    )


if __name__ == "__main__":
    main(parse_args())
