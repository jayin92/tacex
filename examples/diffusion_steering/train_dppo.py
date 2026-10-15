#!/usr/bin/env python
"""Standalone simulation setup for the pi-zero DPPO baseline."""

from __future__ import annotations

import os

from examples.diffusion_steering.sim_path_bootstrap import bootstrap_sim_paths

bootstrap_sim_paths()

# Keep the same XLA inference optimization as the existing simulation runner.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_gpu_triton_gemm_any=True"
)

import pathlib
import tempfile

import jax
import numpy as np
import tensorflow as tf

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from examples.diffusion_steering.train_utils_dppo import trajwise_dppo_training_loop
from examples.diffusion_steering.train_utils_sim import (
    action_bounds_to_config_list,
    get_env_action_bounds,
    infer_actual_action_dim,
    obs_to_agent_input,
    obs_to_img,
)
from jaxrl2.agents.pixel_dppo.ensemble_reward import DPPODisagreementReward
from jaxrl2.agents.pixel_dppo.pi0_policy import create_pi0_dppo_policy
from jaxrl2.agents.pixel_dppo.value import DPPOValueLearner
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from openpi.shared import download
from openpi.training import config as openpi_config


def _resolve_touch_gripper_type(libero_robot, touch_gripper_type):
    if (
            str(libero_robot).endswith("Piper")
            and touch_gripper_type in (
                None, "default", "Robotiq85TactileGripper")):
        return "PiperGripper"
    return touch_gripper_type


def _make_libero_env(
        task,
        resolution,
        seed,
        *,
        libero_robot,
        use_touch,
        touch_gripper_type,
):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "robots": [libero_robot],
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if use_touch:
        env_args["gripper_types"] = touch_gripper_type
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


def _reset_observation(env):
    result = env.reset()
    if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)):
        return result[0]
    return result


def main(variant):
    if variant.algorithm != "pixel_dppo":
        raise ValueError("train_dppo only supports algorithm=pixel_dppo.")
    if variant.env != "libero":
        raise NotImplementedError(
            "The initial isolated DPPO baseline is simulation-only on LIBERO.")
    tf.config.set_visible_devices([], "GPU")
    print(f"DPPO devices: {jax.local_devices()}")

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]
    experiment_name = create_exp_name(
        variant.prefix, seed=variant.seed)
    if variant.suffix:
        experiment_name += f"_{variant.suffix}"
    experiment_root = os.environ.get("EXP", "./experiments")
    variant.outputdir = os.path.join(experiment_root, experiment_name)
    os.makedirs(variant.outputdir, exist_ok=True)
    print(f"Writing DPPO run to {variant.outputdir}")

    benchmark_dict = benchmark.get_benchmark_dict()
    if variant.libero_suite not in benchmark_dict:
        supported = ", ".join(sorted(benchmark_dict))
        raise ValueError(
            f"Unsupported LIBERO suite {variant.libero_suite!r}; "
            f"choose one of: {supported}.")
    task_suite = benchmark_dict[variant.libero_suite]()
    task = task_suite.get_task(int(variant.libero_task_id))
    variant.task_description = task.language
    variant.libero_robot = str(variant.libero_robot)
    variant.touch_gripper_type = _resolve_touch_gripper_type(
        variant.libero_robot, variant.touch_gripper_type)
    env = _make_libero_env(
        task,
        256,
        int(variant.seed),
        libero_robot=variant.libero_robot,
        use_touch=bool(variant.use_touch),
        touch_gripper_type=variant.touch_gripper_type,
    )
    # Evaluation occurs only between complete rollout batches, so sharing the
    # simulator is safe and avoids constructing a second MuJoCo context.
    eval_env = env
    variant.env_max_reward = 1
    variant.max_timesteps = 400
    variant.actual_action_dim = infer_actual_action_dim(variant, env)
    action_low, action_high = get_env_action_bounds(
        env, variant.actual_action_dim)
    variant.actual_action_low = action_bounds_to_config_list(action_low)
    variant.actual_action_high = action_bounds_to_config_list(action_high)
    print(
        f"Using LIBERO task {variant.libero_task_id}: "
        f"{variant.task_description}")
    print(f"Direct DPPO action dimension: {variant.actual_action_dim}")

    group_name = f"{variant.prefix}_{variant.launch_group_id}"
    wandb_logger = WandBLogger(
        variant.prefix != "",
        variant,
        variant.wandb_project,
        experiment_id=experiment_name,
        output_dir=tempfile.mkdtemp(),
        group_name=group_name,
    )

    pi0_config = openpi_config.get_config("pi0_libero")
    checkpoint_dir = download.maybe_download(
        "s3://openpi-assets/checkpoints/pi0_libero")
    policy = create_pi0_dppo_policy(
        pi0_config,
        checkpoint_dir,
        learning_rate=float(variant.dppo_lr),
        trainable_scope=str(variant.dppo_trainable_scope),
        num_denoising_steps=int(variant.dppo_num_denoising_steps),
        finetune_denoising_steps=int(
            variant.dppo_finetune_denoising_steps),
        min_sampling_std=float(variant.dppo_min_sampling_std),
        min_logprob_std=float(variant.dppo_min_logprob_std),
        gamma_denoising=float(variant.dppo_gamma_denoising),
        clip_epsilon=float(variant.dppo_clip_epsilon),
        clip_epsilon_base=float(variant.dppo_clip_epsilon_base),
        clip_epsilon_rate=float(variant.dppo_clip_epsilon_rate),
        max_grad_norm=float(variant.dppo_max_grad_norm),
        target_kl=float(variant.dppo_target_kl),
        normalize_advantages=bool(variant.dppo_normalize_advantages),
        entropy_coefficient=float(variant.dppo_entropy_coef),
        seed=int(variant.seed),
    )
    variant.pi0_action_horizon = policy.action_horizon
    variant.pi0_action_dim = policy.action_dim

    sample_raw_observation = _reset_observation(env)
    sample_agent_observation = obs_to_agent_input(
        sample_raw_observation,
        variant,
        curr_image=obs_to_img(sample_raw_observation, variant),
    )
    hidden_dims = tuple(int(value) for value in variant.hidden_dims)
    if len(hidden_dims) == 1:
        hidden_dims = hidden_dims * 3
    value_learner = DPPOValueLearner(
        int(variant.seed) + 17,
        sample_agent_observation,
        learning_rate=float(variant.dppo_critic_lr),
        max_grad_norm=float(variant.dppo_max_grad_norm),
        value_loss_coefficient=float(variant.dppo_value_loss_coef),
        hidden_dims=hidden_dims,
        cnn_features=tuple(variant.cnn_features),
        cnn_strides=tuple(variant.cnn_strides),
        cnn_padding=str(variant.cnn_padding),
        latent_dim=int(variant.latent_dim),
        encoder_type=str(variant.encoder_type),
        encoder_norm=str(variant.encoder_norm),
        use_spatial_softmax=bool(variant.use_spatial_softmax),
        softmax_temperature=float(variant.softmax_temperature),
        use_bottleneck=bool(variant.use_bottleneck),
        obs_dim=int(variant.obs_dim),
        tactile_hidden_dims=tuple(variant.tactile_hidden_dims),
        mask_touch=bool(variant.mask_touch),
    )
    sample_executed_actions = np.zeros(
        (1, int(variant.query_freq), int(variant.actual_action_dim)),
        dtype=np.float32,
    )
    reward_model = DPPODisagreementReward(
        int(variant.seed) + 29,
        sample_agent_observation,
        sample_executed_actions,
        obs_dim=int(variant.obs_dim),
        model_hidden_dims=tuple(variant.model_hidden_dims),
        num_model_heads=int(variant.num_model_heads),
        model_lr=float(variant.model_lr),
        model_wd=float(variant.model_wd),
        num_update_epochs=int(variant.dppo_ensemble_epochs),
        predict_diff=bool(variant.predict_diff),
        predict_reward=bool(variant.predict_reward),
        ensemble_disagreement_modalities=(
            str(variant.ensemble_disagreement_modalities)),
    )

    trajwise_dppo_training_loop(
        variant,
        policy,
        value_learner,
        reward_model,
        env,
        eval_env,
        wandb_logger,
    )


if __name__ == "__main__":
    raise RuntimeError(
        "Use `python -m examples.diffusion_steering.launch_train_dppo` so flags are parsed.")
