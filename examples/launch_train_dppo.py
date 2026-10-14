"""CLI entrypoint for the isolated pi-zero DPPO experiment."""

import argparse
import sys

from examples.sim_path_bootstrap import bootstrap_sim_paths

bootstrap_sim_paths()

from examples.train_dppo import main
from jaxrl2.utils.general_utils import AttrDict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune pi-zero with stochastic flow-DPPO and ensemble "
            "disagreement reward."))
    parser.add_argument("--algorithm", default="pixel_dppo")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--launch_group_id", default="")
    parser.add_argument("--env", default="libero")
    parser.add_argument("--libero_suite", default="libero_90")
    parser.add_argument("--libero_task_id", default=58, type=int)
    parser.add_argument("--libero_robot", default="Panda")
    parser.add_argument("--prefix", default="pi0_dppo")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--wandb_project", default="DSRL_pi0_DPPO")

    parser.add_argument("--max_steps", default=500_000, type=int)
    parser.add_argument("--max_timesteps", default=400, type=int)
    parser.add_argument("--query_freq", default=20, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--eval_interval", default=10_000, type=int)
    parser.add_argument("--checkpoint_interval", default=10_000, type=int)
    parser.add_argument("--dppo_restore_checkpoint", default="")
    parser.add_argument("--log_interval", default=500, type=int)
    parser.add_argument("--discount", default=0.999, type=float)

    parser.add_argument("--resize_image", default=64, type=int)
    parser.add_argument("--add_states", default=1, type=int)
    parser.add_argument("--add_tactile", default=1, type=int)
    parser.add_argument("--use_touch", default=1, type=int)
    parser.add_argument(
        "--touch_gripper_type", default="PandaGripper")
    parser.add_argument(
        "--tactile_shape", nargs=3, default=(32, 64, 3), type=int)
    parser.add_argument("--gripper_state_dim", default=2, type=int)
    parser.add_argument(
        "--gripper_state_indices", nargs="*", default=None, type=int)
    parser.add_argument("--num_cameras", default=1, type=int)
    parser.add_argument("--action_magnitude", default=1.0, type=float)
    # Accepted only for compatibility with launch metadata. DPPO directly
    # emits robot action chunks and never routes through a DSRL action mode.
    parser.add_argument("--dsrl_action_mode", default="noise")

    parser.add_argument("--hidden_dims", nargs="+", default=(128,), type=int)
    parser.add_argument(
        "--cnn_features", nargs="+", default=(32, 32, 32, 32), type=int)
    parser.add_argument(
        "--cnn_strides", nargs="+", default=(2, 1, 1, 1), type=int)
    parser.add_argument("--cnn_padding", default="VALID")
    parser.add_argument("--latent_dim", default=50, type=int)
    parser.add_argument("--encoder_type", default="small")
    parser.add_argument("--encoder_norm", default="group")
    parser.add_argument("--use_spatial_softmax", default=1, type=int)
    parser.add_argument("--softmax_temperature", default=-1.0, type=float)
    parser.add_argument("--use_bottleneck", default=1, type=int)
    parser.add_argument("--tactile_hidden_dims", nargs="+", default=(256, 256), type=int)
    parser.add_argument("--mask_touch", default=0, type=int)

    parser.add_argument("--model_lr", default=1e-3, type=float)
    parser.add_argument("--model_wd", default=1e-4, type=float)
    parser.add_argument(
        "--model_hidden_dims", nargs="+", default=(256, 256), type=int)
    parser.add_argument("--model_noise_var", default=1.0, type=float)
    parser.add_argument("--num_model_heads", default=5, type=int)
    parser.add_argument("--predict_reward", default=1, type=int)
    parser.add_argument("--predict_diff", default=1, type=int)
    parser.add_argument("--model_obs_key", default="state")
    parser.add_argument("--obs_dim", default=64, type=int)
    parser.add_argument(
        "--ensemble_disagreement_modalities", default="tactile")

    parser.add_argument("--dppo_lr", default=5e-5, type=float)
    parser.add_argument("--dppo_critic_lr", default=1e-3, type=float)
    parser.add_argument(
        "--dppo_trainable_scope",
        default="action_head",
        choices=("action_head", "action_projections", "action_expert", "all"),
    )
    parser.add_argument(
        "--dppo_reward_mode", default="ensemble", choices=("ensemble",))
    parser.add_argument("--dppo_gamma_env", default=0.999, type=float)
    parser.add_argument("--dppo_rollout_steps", default=40, type=int)
    parser.add_argument("--dppo_num_epochs", default=1, type=int)
    parser.add_argument(
        "--dppo_num_minibatches",
        default=0,
        type=int,
        help=(
            "Fallback number of partitions when --dppo_minibatch_size is 0."),
    )
    parser.add_argument(
        "--dppo_minibatch_size",
        default=4,
        type=int,
        help=(
            "Maximum denoising transitions per Pi0 actor update; set to 0 "
            "to partition by --dppo_num_minibatches instead."),
    )
    parser.add_argument("--dppo_clip_epsilon", default=1e-2, type=float)
    parser.add_argument("--dppo_clip_epsilon_base", default=1e-3, type=float)
    parser.add_argument("--dppo_clip_epsilon_rate", default=3.0, type=float)
    parser.add_argument("--dppo_gae_lambda", default=0.95, type=float)
    parser.add_argument("--dppo_value_loss_coef", default=0.5, type=float)
    parser.add_argument("--dppo_critic_warmup_rounds", default=2, type=int)
    parser.add_argument("--dppo_entropy_coef", default=0.0, type=float)
    parser.add_argument("--dppo_max_grad_norm", default=0.5, type=float)
    parser.add_argument("--dppo_target_kl", default=1.0, type=float)
    parser.add_argument("--dppo_normalize_advantages", default=1, type=int)
    parser.add_argument("--dppo_num_denoising_steps", default=10, type=int)
    parser.add_argument(
        "--dppo_finetune_denoising_steps", default=5, type=int)
    parser.add_argument("--dppo_gamma_denoising", default=0.9, type=float)
    parser.add_argument("--dppo_min_sampling_std", default=0.1, type=float)
    parser.add_argument("--dppo_min_logprob_std", default=0.1, type=float)
    parser.add_argument("--dppo_reward_scale_running", default=1, type=int)
    parser.add_argument("--dppo_ensemble_epochs", default=10, type=int)
    parser.add_argument(
        "--dppo_intrinsic_reward_scale", default=1.0, type=float)
    parser.add_argument(
        "--dppo_extrinsic_reward_scale", default=0.0, type=float)
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.algorithm != "pixel_dppo":
        raise ValueError(
            "The isolated DPPO entrypoint only accepts --algorithm pixel_dppo.")
    return AttrDict(vars(args))


if __name__ == "__main__":
    main(parse_args())
    sys.exit()
