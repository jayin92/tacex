"""Launch DPPO on the seven requested tasks remaining after task 58.

This launcher is intentionally separate from the DSRL-PPO and SAC launchers so
that adding the baseline cannot change their experiment grids. Task 58 was
launched separately; this grid keeps the same project and configuration while
launching each remaining LIBERO task over ten seeds.
"""

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


ENTITY = "kiten"
PROJECT_NAME = "DSRL_pi0_Libero_DPPO_Disagreement_Task58"
MODULE_NAME = "examples.launch_train_dppo"

LIBERO_ROBOT = "Panda"
TOUCH_GRIPPER_TYPE = "PandaGripper"
SEEDS = tuple(range(10))
TASK_IDS = (69, 61, 77, 55, 21, 56, 20)


def _int_list_from_env(name, default):
    value = os.environ.get(name, default)
    return [int(item) for item in value.replace(",", " ").split()]


# Keep the environment, tactile observation, ensemble model, and W&B project
# identical to the already-launched task-58 condition. DPPO receives ensemble
# disagreement as its query-level reward; sparse task reward is disabled.
BASE_FLAGS = {
    "algorithm": "pixel_dppo",
    "env": "libero",
    "libero_robot": LIBERO_ROBOT,
    "prefix": "pi0_libero_flow_dppo_disagreement",
    "wandb_project": PROJECT_NAME,
    "max_steps": 500000,
    "eval_interval": 10000,
    "checkpoint_interval": 10000,
    "eval_episodes": 10,
    "resize_image": 64,
    "add_tactile": 1,
    "use_touch": 1,
    "touch_gripper_type": TOUCH_GRIPPER_TYPE,
    "tactile_shape": _int_list_from_env("tactile_shape", "32 64 3"),
    "gripper_state_indices": [0, 1],
    "query_freq": 20,
    "hidden_dims": [128],
    "model_lr": 1e-3,
    "model_wd": 1e-4,
    "model_hidden_dims": [256, 256],
    "num_model_heads": 5,
    "predict_reward": 1,
    "predict_diff": 1,
    "obs_dim": 64,
    "ensemble_disagreement_modalities": "tactile",
    "tactile_hidden_dims": [256, 256],
    "mask_touch": 0,
    # DPPO-informed defaults for the stochastic pi-zero flow adaptation. A
    # rollout is expanded across the five fine-tuned flow transitions before
    # minibatching.
    "dppo_lr": 5e-5,
    "dppo_critic_lr": 1e-3,
    "dppo_trainable_scope": "action_head",
    "dppo_reward_mode": "ensemble",
    "dppo_gamma_env": 0.999,
    # Pi0 is substantially larger than the diffusion backbones in the paper.
    # These conservative defaults size the baseline for one RTX 4090;
    # increase them only after profiling the target node.
    "dppo_rollout_steps": 40,
    "dppo_num_epochs": 1,
    # A full Pi0 forward pass is required for each PPO minibatch. Capping the
    # denoising-sample batch keeps the action-head update viable on a 4090.
    "dppo_minibatch_size": 4,
    "dppo_clip_epsilon": 1e-2,
    "dppo_gae_lambda": 0.95,
    "dppo_value_loss_coef": 0.5,
    "dppo_critic_warmup_rounds": 2,
    "dppo_entropy_coef": 0.0,
    "dppo_max_grad_norm": 0.5,
    "dppo_target_kl": 1.0,
    "dppo_normalize_advantages": 1,
    "dppo_num_denoising_steps": 10,
    "dppo_finetune_denoising_steps": 5,
    "dppo_gamma_denoising": 0.9,
    "dppo_min_sampling_std": 0.1,
    "dppo_min_logprob_std": 0.1,
    "dppo_reward_scale_running": 1,
    "dppo_ensemble_epochs": 10,
    "dppo_intrinsic_reward_scale": 1.0,
    "dppo_extrinsic_reward_scale": 0.0,
}


SWEEP_FLAGS = {
    "seed": list(SEEDS),
    "libero_suite": ["libero_90"],
    "libero_task_id": list(TASK_IDS),
}


def build_flags(project_name=PROJECT_NAME):
    for sweep_flags in dict_permutations(SWEEP_FLAGS):
        flags = copy.deepcopy(BASE_FLAGS)
        flags["wandb_project"] = project_name
        flags.update(sweep_flags)
        flags["suffix"] = hash_dict(sweep_flags)
        yield flags


def slug(value):
    value = value.replace(".", "p").replace(",", "-")
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")


def job_name(flags):
    seed = flags.get("seed", "no_seed")
    task_id = flags.get("libero_task_id", "no_task")
    suffix = flags.get("suffix", "run")
    return slug(f"{flags['prefix']}_task{task_id}_{suffix}_seed{seed}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Launch ten seeds for each of the seven DPPO tasks remaining after "
            "task 58."
        )
    )
    parser.add_argument("--project_name", type=str, default=PROJECT_NAME)
    parser.add_argument("--entity", type=str, default=ENTITY)
    parser.add_argument("--scratch_root", type=str, default="/cluster/scratch")
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu_type", type=str, default="rtx_4090")
    parser.add_argument("--mem", type=int, default=32000)
    parser.add_argument("--duration", type=str, default="23:59:00")
    parser.add_argument(
        "--mode",
        type=str,
        default="euler",
        choices=("euler", "local", "local_async"),
    )
    parser.add_argument("--dry", default=False, action="store_true")
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

    command_list = []
    output_file_list = []
    for flags in build_flags(args.project_name):
        command_list.append(
            generate_base_command(
                MODULE_NAME,
                flags=flags,
                env=env,
                pre_commands=pre_commands,
            )
        )
        output_file_list.append(
            os.path.join(slurm_dir, f"{job_name(flags)}-%j.out")
        )

    print(f"Prepared {len(command_list)} DPPO {args.mode} jobs.")
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
