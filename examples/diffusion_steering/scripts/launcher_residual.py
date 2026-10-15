"""Launch a single residual-RL baseline on LIBERO task 58 with Panda.

The default profile is the full 500k-update experiment. Pass ``--smoke`` to
exercise environment collection, pi0 inference, SAC compilation, and evaluation
with a much shorter run.
"""

import argparse
import os
import re
import shlex
from pathlib import Path

from jaxrl2.utils.launch_util import generate_base_command, generate_run_commands


ENTITY = "kiten"
PROJECT_NAME = "DSRL_pi0_Libero_Residual_Panda_Task58"
MODULE_NAME = "examples.diffusion_steering.launch_train_sim"
REPO_ROOT = Path(__file__).resolve().parents[3]


FULL_FLAGS = {
    "algorithm": "pixel_sac",
    "env": "libero",
    "libero_suite": "libero_90",
    "libero_task_id": 58,
    "libero_robot": "Panda",
    # Classic residual RL: learn only a 7-D correction to pi0's actions.
    "dsrl_action_mode": "delta",
    "residual_delta_fraction": 0.1,
    "batch_size": 256,
    "discount": 0.999,
    "seed": 0,
    "max_steps": 500_000,
    "checkpoint_interval": 20_000,
    "eval_interval": 20_000,
    "log_interval": 1_000,
    "eval_episodes": 10,
    "multi_grad_step": 20,
    "start_online_updates": 500,
    "resize_image": 64,
    "add_states": 1,
    # Match the existing Panda task-58 setup without feeding touch to SAC.
    "add_tactile": 0,
    "use_touch": 1,
    "touch_gripper_type": "PandaGripper",
    "gripper_state_indices": [0, 1],
    "action_magnitude": 1.0,
    "query_freq": 20,
    "hidden_dims": [128],
}


SMOKE_OVERRIDES = {
    "batch_size": 64,
    "max_steps": 1_000,
    "checkpoint_interval": 1_000,
    "eval_interval": 1_000,
    "log_interval": 100,
    "eval_episodes": 1,
    "multi_grad_step": 20,
    # One failed 400-step rollout contains 20 query transitions. Since the
    # training condition is strict (>), this starts updates after rollout 2.
    "start_online_updates": 20,
}


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _slug(value):
    value = str(value).replace(".", "p").replace(",", "-")
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")


def build_flags(args):
    flags = dict(FULL_FLAGS)
    if args.smoke:
        flags.update(SMOKE_OVERRIDES)

    profile_suffix = "_smoke" if args.smoke else ""
    flags.update(
        seed=args.seed,
        dsrl_action_mode=args.action_mode,
        residual_delta_fraction=args.residual_delta_fraction,
        prefix=f"residual_rl_pi0_panda_task58_{args.action_mode}{profile_suffix}",
        wandb_project=args.project_name,
    )
    if args.max_steps is not None:
        flags["max_steps"] = args.max_steps
    if args.suffix:
        flags["suffix"] = args.suffix
    return flags


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Launch one Panda residual-RL run on LIBERO-90 task 58 "
            "(pick up the ketchup and put it in the tray)."
        )
    )
    parser.add_argument("--smoke", action="store_true",
                        help="run the short compilation/integration profile")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action_mode", choices=("delta", "both"),
                        default="delta",
                        help="delta is the classic residual-only baseline")
    parser.add_argument("--residual_delta_fraction", type=_positive_float,
                        default=0.1)
    parser.add_argument("--max_steps", type=_positive_int, default=None,
                        help="override the selected profile's update budget")
    parser.add_argument("--suffix", default="",
                        help="optional run-name suffix")

    parser.add_argument("--project_name", default=PROJECT_NAME)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--scratch_root", default="/cluster/scratch")
    parser.add_argument("--exp_dir", default=None)
    parser.add_argument(
        "--openpi_data_home",
        default=str(REPO_ROOT / "openpi"),
        help="persistent cache containing the pi0_libero checkpoint",
    )
    parser.add_argument("--mode", choices=("euler", "local", "local_async"),
                        default="euler")
    parser.add_argument("--num_cpus", type=_positive_int, default=1)
    parser.add_argument("--num_gpus", type=_positive_int, default=1)
    parser.add_argument("--gpu_type", default="rtx_4090")
    parser.add_argument("--mem", type=_positive_int, default=32_000)
    parser.add_argument("--duration", default=None)
    parser.add_argument("--dry", action="store_true",
                        help="print the generated command without submitting")
    parser.add_argument("--yes", action="store_true",
                        help="submit without the confirmation prompt")
    return parser.parse_args()


def main(args):
    exp_dir = args.exp_dir or os.path.join(
        args.scratch_root, args.entity, args.project_name
    )
    slurm_dir = os.path.join(exp_dir, "slurm")
    if not args.dry:
        os.makedirs(slurm_dir, exist_ok=True)

    flags = build_flags(args)
    profile = "smoke" if args.smoke else "full"
    job_name = _slug(
        f"residual-panda-task58-{args.action_mode}-{profile}-seed{args.seed}"
    )
    pre_commands = [
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"mkdir -p {shlex.quote(exp_dir)} {shlex.quote(slurm_dir)}",
    ]
    env = {
        "EXP": exp_dir,
        "OPENPI_DATA_HOME": args.openpi_data_home,
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MPLCONFIGDIR": os.path.join(exp_dir, "matplotlib"),
    }
    command = generate_base_command(
        MODULE_NAME,
        flags=flags,
        env=env,
        pre_commands=pre_commands,
    )

    duration = args.duration or ("3:59:00" if args.smoke else "23:59:00")
    print(
        f"Prepared 1 {args.mode} {profile} job: Panda, libero_90/task 58, "
        f"action_mode={args.action_mode}, seed={args.seed}."
    )
    generate_run_commands(
        [command],
        output_file_list=[os.path.join(slurm_dir, f"{job_name}-%j.out")],
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        dry=args.dry,
        mem=args.mem,
        duration=duration,
        mode=args.mode,
        prompt=not (args.yes or args.dry),
        gpu_type=args.gpu_type,
    )


if __name__ == "__main__":
    main(parse_args())
