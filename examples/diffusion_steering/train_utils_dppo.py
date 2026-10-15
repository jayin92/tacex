"""On-policy training loop for stochastic pi-zero DPPO.

This path is independent of the replay-based DSRL loops and of DSRL-PPO.  A
rollout stores pi-zero's selected denoising chain, assigns one ensemble-
disagreement reward per environment query, computes environment-level GAE, and
then expands each query across its fine-tuned denoising transitions for PPO.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
import pickle
import tempfile
import time
from typing import Any

import jax
import numpy as np
from tqdm import tqdm
import wandb

from examples.diffusion_steering.train_utils_sim import (
    apply_residual_delta,
    obs_to_agent_input,
    obs_to_img,
    obs_to_pi_zero_input,
    obs_to_tactile,
)


def compute_dppo_gae(
        rewards: np.ndarray,
        values: np.ndarray,
        bootstrap_value: float,
        masks: np.ndarray,
        gamma_env: float,
        gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes GAE once per policy query, as in DPPO.

    ``gamma_env`` is a per-query discount.  It is intentionally not raised to
    ``query_freq``: the official DPPO implementation treats one action chunk as
    one outer-MDP transition.  A zero mask denotes a true termination, while a
    time-limit or rollout truncation retains a mask of one and bootstraps.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    if not (
            rewards.ndim == values.ndim == masks.ndim == 1
            and len(rewards) == len(values) == len(masks)):
        raise ValueError(
            "rewards, values, and masks must be equal-length 1D arrays.")
    if not 0.0 < float(gamma_env) <= 1.0:
        raise ValueError("gamma_env must be in (0, 1].")
    if not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")

    advantages = np.empty_like(rewards)
    next_value = np.float32(bootstrap_value)
    next_advantage = np.float32(0.0)
    gamma = np.float32(gamma_env)
    lam = np.float32(gae_lambda)
    for index in range(len(rewards) - 1, -1, -1):
        continuation = gamma * masks[index]
        delta = rewards[index] + continuation * next_value - values[index]
        next_advantage = delta + continuation * lam * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values


class RunningRewardScale:
    """Divides rewards by a running standard deviation without centering."""

    def __init__(self, epsilon: float = 1e-4):
        self.count = float(epsilon)
        self.mean = 0.0
        self.m2 = float(epsilon)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.m2 / self.count, 1e-8)))

    def update(self, values: np.ndarray) -> None:
        for value in np.asarray(values, dtype=np.float64).reshape(-1):
            self.count += 1.0
            delta = float(value) - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (float(value) - self.mean)

    def scale(
            self,
            rewards: np.ndarray,
            *,
            statistics_values: np.ndarray | None = None,
    ) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float32)
        values = rewards if statistics_values is None else statistics_values
        self.update(values)
        return np.clip(rewards / self.std, -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: Mapping[str, float]) -> None:
        self.count = float(state["count"])
        self.mean = float(state["mean"])
        self.m2 = float(state["m2"])


def _reset_env(env: Any, variant: Any):
    result = env.reset()
    if "aloha" in variant.env:
        return result[0]
    if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], Mapping)):
        return result[0]
    return result


def _step_env(env: Any, action: np.ndarray):
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info or {},
        )
    if len(result) == 4:
        observation, reward, done, info = result
        info = info or {}
        truncated = bool(
            info.get("TimeLimit.truncated", False)
            or info.get("truncated", False))
        return (
            observation,
            float(reward),
            bool(done and not truncated),
            truncated,
            info,
        )
    raise ValueError(
        f"Expected env.step() to return 4 or 5 values, got {len(result)}.")


def _agent_observation(observation: Mapping[str, Any], variant: Any):
    image = obs_to_img(observation, variant)
    return obs_to_agent_input(
        observation, variant, curr_image=image), image


def _has_contact(observation: Mapping[str, Any], variant: Any) -> bool:
    if not bool(getattr(variant, "use_touch", 0)):
        return False
    try:
        tactile = obs_to_tactile(observation, variant)
    except (KeyError, NotImplementedError):
        return False
    return bool(np.any(np.abs(tactile) > 1e-8))


def _stack_agent_observations(
        observations: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not observations:
        raise ValueError("At least one observation is required.")
    keys = tuple(observations[0])
    if any(tuple(observation) != keys for observation in observations):
        raise ValueError("All agent observations must have identical keys.")
    return {
        key: np.concatenate(
            [np.asarray(observation[key]) for observation in observations],
            axis=0,
        )
        for key in keys
    }


def _stack_model_observations(observations: Sequence[Any]):
    if not observations:
        raise ValueError("At least one pi-zero observation is required.")

    def concatenate(*values):
        if values[0] is None:
            return None
        return np.concatenate(
            [np.asarray(value) for value in values], axis=0)

    return jax.tree.map(
        concatenate,
        *observations,
        is_leaf=lambda value: value is None,
    )


def _tree_take(tree, indices):
    return jax.tree.map(
        lambda value: None if value is None else value[indices],
        tree,
        is_leaf=lambda value: value is None,
    )


def collect_dppo_trajectory(
        variant: Any,
        policy: Any,
        value_learner: Any,
        env: Any,
        *,
        max_env_steps: int | None = None,
) -> dict[str, Any]:
    """Collects one fresh on-policy DPPO episode."""
    query_frequency = int(variant.query_freq)
    if query_frequency <= 0:
        raise ValueError("query_freq must be positive.")
    episode_limit = int(variant.max_timesteps)
    if max_env_steps is not None:
        episode_limit = min(episode_limit, int(max_env_steps))
    if episode_limit <= 0:
        raise ValueError("The remaining environment budget must be positive.")

    actual_action_dim = int(variant.actual_action_dim)
    observation = _reset_env(env, variant)
    agent_observations = []
    next_agent_observations = []
    model_observations = []
    chains = []
    old_log_probs = []
    old_values = []
    executed_action_chunks = []
    coordinate_masks = []
    query_lengths = []
    extrinsic_rewards = []
    images = []

    total_steps = 0
    contact_steps = 0
    terminated = False
    env_truncated = False
    success_seen = False

    while total_steps < episode_limit and not (terminated or env_truncated):
        agent_observation, current_image = _agent_observation(
            observation, variant)
        pi0_observation = policy.prepare_observation(
            obs_to_pi_zero_input(observation, variant))
        sample = policy.sample(pi0_observation, deterministic=False)
        actions = np.asarray(sample["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != actual_action_dim:
            raise ValueError(
                "DPPO pi-zero output must have shape (horizon, actual_action_dim); "
                f"expected second dimension {actual_action_dim}, got {actions.shape}.")
        if len(actions) < query_frequency:
            raise ValueError(
                f"query_freq={query_frequency} exceeds pi-zero action horizon "
                f"{len(actions)}.")
        actions = apply_residual_delta(
            variant, actions, delta_normalized=None)

        value = float(np.asarray(
            value_learner.value(agent_observation)).reshape(-1)[0])
        agent_observations.append(agent_observation)
        model_observations.append(pi0_observation)
        chains.append(np.asarray(sample["chains"])[0])
        old_log_probs.append(np.asarray(sample["old_log_probs"])[0])
        old_values.append(value)

        remaining = episode_limit - total_steps
        planned = min(query_frequency, remaining)
        executed = 0
        query_reward = 0.0
        action_chunk = np.zeros(
            (query_frequency, actual_action_dim), dtype=np.float32)
        for action_index in range(planned):
            if _has_contact(observation, variant):
                contact_steps += 1
            action = actions[action_index]
            observation, reward, terminated, env_truncated, _ = _step_env(
                env, action)
            images.append(current_image)
            action_chunk[action_index] = action
            query_reward += reward
            success_seen = (
                success_seen
                or reward == float(variant.env_max_reward))
            total_steps += 1
            executed += 1
            if terminated or env_truncated:
                break

        next_agent_observation, _ = _agent_observation(observation, variant)
        next_agent_observations.append(next_agent_observation)
        executed_action_chunks.append(action_chunk)
        query_lengths.append(executed)
        extrinsic_rewards.append(query_reward)

        coordinate_mask = np.zeros(
            (policy.action_horizon, policy.action_dim), dtype=bool)
        coordinate_mask[:executed, :actual_action_dim] = True
        coordinate_masks.append(coordinate_mask)

    budget_truncated = total_steps >= episode_limit and not terminated
    truncated = bool(env_truncated or budget_truncated)
    final_agent_observation, final_image = _agent_observation(
        observation, variant)
    images.append(final_image)

    masks = np.ones((len(chains),), dtype=np.float32)
    if terminated:
        masks[-1] = 0.0
    bootstrap_value = 0.0
    if masks[-1] != 0.0:
        bootstrap_value = float(np.asarray(
            value_learner.value(final_agent_observation)).reshape(-1)[0])

    is_success = bool(success_seen)
    episode_return = float(np.sum(extrinsic_rewards))
    print(
        "DPPO rollout done: "
        f"return={episode_return}, success={is_success}, "
        f"queries={len(chains)}, env_steps={total_steps}, "
        f"terminated={terminated}, truncated={truncated}")
    return {
        "observations": agent_observations,
        "next_observations": next_agent_observations,
        "model_observations": model_observations,
        "chains": np.asarray(chains, dtype=np.float32),
        "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
        "old_values": np.asarray(old_values, dtype=np.float32),
        "bootstrap_value": np.float32(bootstrap_value),
        "executed_actions": np.asarray(
            executed_action_chunks, dtype=np.float32),
        "coordinate_masks": np.asarray(coordinate_masks, dtype=bool),
        "query_lengths": np.asarray(query_lengths, dtype=np.int32),
        "real_action_dim": np.full(
            (len(chains),), actual_action_dim, dtype=np.int32),
        "extrinsic_rewards": np.asarray(extrinsic_rewards, dtype=np.float32),
        "masks": masks,
        "terminated": terminated,
        "truncated": truncated,
        "is_success": is_success,
        "episode_return": episode_return,
        "env_steps": total_steps,
        "contact_steps": contact_steps,
        "contact_frequency": contact_steps / max(total_steps, 1),
        "images": images,
    }


def assign_ensemble_rewards(
        trajectories: Sequence[dict[str, Any]],
        reward_model: Any,
        ensemble_state: Any,
        *,
        intrinsic_scale: float,
        extrinsic_scale: float,
        gamma_env: float = 0.999,
        running_reward_scale: RunningRewardScale | None = None,
) -> tuple[Any, dict[str, float]]:
    """Scores all rollout transitions before one ensemble fit."""
    observations = _stack_agent_observations([
        observation
        for trajectory in trajectories
        for observation in trajectory["observations"]
    ])
    next_observations = _stack_agent_observations([
        observation
        for trajectory in trajectories
        for observation in trajectory["next_observations"]
    ])
    actions = np.concatenate(
        [trajectory["executed_actions"] for trajectory in trajectories],
        axis=0,
    )
    extrinsic = np.concatenate(
        [trajectory["extrinsic_rewards"] for trajectory in trajectories],
        axis=0,
    )
    intrinsic, ensemble_state, metrics = reward_model.score_and_update(
        observations,
        actions,
        next_observations,
        extrinsic,
        state=ensemble_state,
    )
    intrinsic = np.asarray(jax.device_get(intrinsic), dtype=np.float32)
    combined = (
        np.float32(intrinsic_scale) * intrinsic
        + np.float32(extrinsic_scale) * extrinsic
    )
    if running_reward_scale is not None:
        # DPPO scales immediate rewards by the running standard deviation of
        # discounted query returns (without subtracting their mean).
        discounted_returns = []
        offset = 0
        for trajectory in trajectories:
            count = len(trajectory["chains"])
            trajectory_rewards = combined[offset:offset + count]
            trajectory_masks = np.asarray(
                trajectory["masks"], dtype=np.float32)
            running_return = np.float32(0.0)
            statistics = np.empty_like(trajectory_rewards)
            for index in range(count - 1, -1, -1):
                running_return = (
                    trajectory_rewards[index]
                    + np.float32(gamma_env)
                    * trajectory_masks[index]
                    * running_return
                )
                statistics[index] = running_return
            discounted_returns.append(statistics)
            offset += count
        combined = running_reward_scale.scale(
            combined,
            statistics_values=np.concatenate(discounted_returns, axis=0),
        )

    offset = 0
    for trajectory in trajectories:
        count = len(trajectory["chains"])
        trajectory["intrinsic_rewards"] = intrinsic[offset:offset + count]
        trajectory["rewards"] = combined[offset:offset + count]
        offset += count

    scalar_metrics = {
        key: float(np.asarray(jax.device_get(value)))
        for key, value in metrics.items()
    }
    scalar_metrics.update({
        "intrinsic_reward_mean": float(np.mean(intrinsic)),
        "intrinsic_reward_std": float(np.std(intrinsic)),
        "combined_reward_mean": float(np.mean(combined)),
        "combined_reward_std": float(np.std(combined)),
    })
    if running_reward_scale is not None:
        scalar_metrics["reward_running_std"] = running_reward_scale.std
    return ensemble_state, scalar_metrics


def make_dppo_batch(
        trajectories: Sequence[Mapping[str, Any]],
        *,
        gamma_env: float,
        gae_lambda: float,
) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("At least one trajectory is required.")

    observations = []
    model_observations = []
    parts = {
        "chains": [],
        "old_log_probs": [],
        "old_values": [],
        "advantages": [],
        "returns": [],
        "query_lengths": [],
        "real_action_dim": [],
        "coordinate_masks": [],
    }
    for trajectory in trajectories:
        advantages, returns = compute_dppo_gae(
            trajectory["rewards"],
            trajectory["old_values"],
            trajectory["bootstrap_value"],
            trajectory["masks"],
            gamma_env,
            gae_lambda,
        )
        observations.extend(trajectory["observations"])
        model_observations.extend(trajectory["model_observations"])
        for key in (
                "chains", "old_log_probs", "old_values",
                "query_lengths", "real_action_dim", "coordinate_masks"):
            parts[key].append(np.asarray(trajectory[key]))
        parts["advantages"].append(advantages)
        parts["returns"].append(returns)

    return {
        "observations": _stack_agent_observations(observations),
        "model_observations": _stack_model_observations(model_observations),
        **{
            key: np.concatenate(values, axis=0)
            for key, values in parts.items()
        },
    }


def update_dppo_batch(
        policy: Any,
        value_learner: Any,
        batch: Mapping[str, Any],
        *,
        num_epochs: int,
        num_minibatches: int,
        minibatch_size: int | None = None,
        update_actor: bool = True,
        seed: int,
) -> dict[str, float]:
    """Runs shuffled PPO epochs over ``query x denoising_step`` samples."""
    query_count = len(batch["advantages"])
    denoising_steps = int(policy.finetune_denoising_steps)
    total_samples = query_count * denoising_steps
    if total_samples <= 0:
        raise ValueError("DPPO update batch must contain transitions.")
    if num_epochs <= 0:
        raise ValueError("DPPO epochs must be positive.")
    if minibatch_size is not None and minibatch_size <= 0:
        raise ValueError("DPPO minibatch_size must be positive when provided.")
    if minibatch_size is None and num_minibatches <= 0:
        raise ValueError("DPPO num_minibatches must be positive.")

    rng = np.random.default_rng(seed)
    actor_advantages = np.asarray(batch["advantages"], dtype=np.float32)
    if bool(getattr(policy, "normalize_advantages", False)):
        # Normalize once over the fresh rollout. With the small Pi0 actor
        # minibatches required by a 4090, per-minibatch normalization can turn
        # a repeated query's advantages entirely into zeros.
        actor_advantages = (
            (actor_advantages - actor_advantages.mean())
            / (actor_advantages.std() + 1e-8)
        )
    collected_metrics: dict[str, list[float]] = {}
    stopped_early = False
    completed_minibatches = 0
    for _ in range(num_epochs):
        permutation = rng.permutation(total_samples)
        if minibatch_size is None:
            epoch_minibatches = np.array_split(
                permutation, num_minibatches)
        else:
            epoch_minibatches = (
                permutation[start:start + minibatch_size]
                for start in range(0, total_samples, minibatch_size)
            )
        for flat_indices in epoch_minibatches:
            if len(flat_indices) == 0:
                continue
            query_indices = flat_indices // denoising_steps
            denoising_indices = flat_indices % denoising_steps
            actor_info = {}
            if update_actor:
                actor_info = policy.update(
                    _tree_take(batch["model_observations"], query_indices),
                    batch["chains"][query_indices, denoising_indices],
                    batch["chains"][query_indices, denoising_indices + 1],
                    denoising_indices.astype(np.int32),
                    batch["old_log_probs"][query_indices, denoising_indices],
                    actor_advantages[query_indices],
                    batch["query_lengths"][query_indices],
                    batch["real_action_dim"][query_indices],
                    batch["coordinate_masks"][query_indices],
                )
            value_info = value_learner.update(
                _tree_take(batch["observations"], query_indices),
                batch["returns"][query_indices],
            )
            metrics = {
                **{f"actor_{key}": value
                   for key, value in actor_info.items()},
                **value_info,
            }
            for key, value in metrics.items():
                collected_metrics.setdefault(key, []).append(float(value))
            completed_minibatches += 1
            if (
                    update_actor
                    and policy.target_kl > 0
                    and actor_info["approx_kl"] > policy.target_kl):
                stopped_early = True
                break
        if stopped_early:
            break

    result = {
        key: float(np.mean(values))
        for key, values in collected_metrics.items()
    }
    result["completed_minibatches"] = float(completed_minibatches)
    result["early_stop"] = float(stopped_early)
    result["actor_updates_enabled"] = float(update_actor)
    result["denoising_samples"] = float(total_samples)
    old_values = np.asarray(batch["old_values"])
    returns = np.asarray(batch["returns"])
    return_variance = float(np.var(returns))
    result["explained_variance"] = (
        float("nan") if return_variance == 0.0
        else 1.0 - float(np.var(returns - old_values)) / return_variance
    )
    return result


def evaluate_dppo_policy(
        policy: Any,
        env: Any,
        variant: Any,
        wandb_logger: Any,
        step: int,
) -> dict[str, float]:
    """Evaluates the updated pi-zero model directly (no DSRL steering)."""
    returns = []
    successes = []
    lengths = []
    for rollout_index in range(int(variant.eval_episodes)):
        observation = _reset_env(env, variant)
        images = []
        episode_rewards = []
        terminated = truncated = False
        env_steps = 0
        while (
                env_steps < int(variant.max_timesteps)
                and not (terminated or truncated)):
            pi0_observation = policy.prepare_observation(
                obs_to_pi_zero_input(observation, variant))
            sample = policy.sample(pi0_observation, deterministic=True)
            actions = apply_residual_delta(
                variant,
                np.asarray(sample["actions"], dtype=np.float32),
                delta_normalized=None,
            )
            for action in actions[:int(variant.query_freq)]:
                images.append(obs_to_img(observation, variant))
                observation, reward, terminated, truncated, _ = _step_env(
                    env, action)
                episode_rewards.append(reward)
                env_steps += 1
                if (
                        terminated
                        or truncated
                        or env_steps >= int(variant.max_timesteps)):
                    break

        episode_return = float(np.sum(episode_rewards))
        success = bool(
            episode_rewards
            and np.max(episode_rewards) >= float(variant.env_max_reward))
        returns.append(episode_return)
        successes.append(float(success))
        lengths.append(env_steps)
        print(
            f"DPPO eval {rollout_index}: return={episode_return}, "
            f"success={success}, length={env_steps}")
        if images:
            video = np.stack(images).transpose(0, 3, 1, 2)
            wandb_logger.log({
                f"eval_video/{rollout_index}": wandb.Video(video, fps=50)
            }, step=step)

    metrics = {
        "evaluation/avg_return": float(np.mean(returns)),
        "evaluation/success_rate": float(np.mean(successes)),
        "evaluation/avg_episode_len": float(np.mean(lengths)),
    }
    wandb_logger.log(metrics, step=step)
    return metrics


def _crossed_interval(previous: int, current: int, interval: int) -> bool:
    return interval > 0 and previous // interval < current // interval


def save_dppo_checkpoint(
        output_dir: str,
        step: int,
        *,
        policy: Any,
        value_learner: Any,
        ensemble_state: Any,
        reward_scale: RunningRewardScale | None,
        counters: Mapping[str, int],
) -> str:
    """Atomically saves every mutable DPPO subsystem."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"dppo_checkpoint_{step}.pkl")
    payload = {
        "step": int(step),
        "policy": policy.state_dict(),
        "value": value_learner.state_dict(),
        "ensemble_state": ensemble_state,
        "reward_scale": (
            None if reward_scale is None else reward_scale.state_dict()),
        "counters": dict(counters),
    }
    payload = jax.device_get(payload)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".dppo-checkpoint-", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(descriptor, "wb") as checkpoint_file:
            pickle.dump(payload, checkpoint_file, protocol=pickle.HIGHEST_PROTOCOL)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return path


def load_dppo_checkpoint(
        checkpoint_path: str,
        *,
        policy: Any,
        value_learner: Any,
        reward_scale: RunningRewardScale | None,
) -> tuple[Any, dict[str, int]]:
    """Restores all mutable learner state from a trusted local checkpoint."""
    with open(checkpoint_path, "rb") as checkpoint_file:
        payload = pickle.load(checkpoint_file)
    required = {
        "policy", "value", "ensemble_state", "reward_scale", "counters"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"DPPO checkpoint is missing required fields: {sorted(missing)}")

    policy.load_state_dict(payload["policy"])
    value_learner.load_state_dict(payload["value"])
    saved_reward_scale = payload["reward_scale"]
    if reward_scale is None and saved_reward_scale is not None:
        raise ValueError(
            "Checkpoint contains running reward-scale state, but reward "
            "scaling is disabled in the current configuration.")
    if reward_scale is not None and saved_reward_scale is None:
        raise ValueError(
            "Checkpoint has no running reward-scale state, but reward "
            "scaling is enabled in the current configuration.")
    if reward_scale is not None:
        reward_scale.load_state_dict(saved_reward_scale)

    counters = {
        key: int(value) for key, value in payload["counters"].items()
    }
    return payload["ensemble_state"], counters


def trajwise_dppo_training_loop(
        variant: Any,
        policy: Any,
        value_learner: Any,
        reward_model: Any,
        env: Any,
        eval_env: Any,
        wandb_logger: Any,
) -> None:
    """Runs the isolated ensemble-reward pi-zero DPPO experiment."""
    rollout_steps = int(variant.dppo_rollout_steps)
    max_env_steps = int(variant.max_steps)
    if rollout_steps <= 0 or max_env_steps <= 0:
        raise ValueError("DPPO rollout_steps and max_steps must be positive.")
    if str(variant.dppo_reward_mode) != "ensemble":
        raise ValueError(
            "This baseline intentionally supports dppo_reward_mode=ensemble only.")

    ensemble_state = reward_model.initial_state
    reward_scale = (
        RunningRewardScale()
        if bool(variant.dppo_reward_scale_running)
        else None
    )
    counters = {
        "total_env_steps": 0,
        "total_queries": 0,
        "total_trajectories": 0,
        "update_round": 0,
    }
    restore_path = str(
        getattr(variant, "dppo_restore_checkpoint", "") or "").strip()
    if restore_path:
        ensemble_state, counters = load_dppo_checkpoint(
            restore_path,
            policy=policy,
            value_learner=value_learner,
            reward_scale=reward_scale,
        )
        print(
            "Restored DPPO checkpoint "
            f"{restore_path} at env step {counters['total_env_steps']}.")
    total_env_steps = counters["total_env_steps"]
    total_queries = counters["total_queries"]
    total_trajectories = counters["total_trajectories"]
    update_round = counters["update_round"]
    session_start_env_steps = total_env_steps
    start_time = time.monotonic()

    wandb_logger.log({
        "env_steps": total_env_steps,
        "num_online_samples": total_queries,
        "num_online_trajs": total_trajectories,
        "dppo/update_round": update_round,
    }, step=total_env_steps)
    evaluate_dppo_policy(
        policy, eval_env, variant, wandb_logger, step=total_env_steps)

    with tqdm(total=max_env_steps, initial=total_env_steps) as progress:
        while total_env_steps < max_env_steps:
            previous_env_steps = total_env_steps
            trajectories = []
            rollout_queries = 0
            rollout_successes = []
            rollout_returns = []
            contact_frequencies = []
            while (
                    rollout_queries < rollout_steps
                    and total_env_steps < max_env_steps):
                trajectory = collect_dppo_trajectory(
                    variant,
                    policy,
                    value_learner,
                    env,
                    max_env_steps=min(
                        max_env_steps - total_env_steps,
                        (
                            rollout_steps - rollout_queries
                        ) * int(variant.query_freq),
                    ),
                )
                trajectories.append(trajectory)
                query_count = len(trajectory["chains"])
                rollout_queries += query_count
                total_queries += query_count
                total_env_steps += int(trajectory["env_steps"])
                total_trajectories += 1
                rollout_successes.append(float(trajectory["is_success"]))
                rollout_returns.append(float(trajectory["episode_return"]))
                contact_frequencies.append(
                    float(trajectory["contact_frequency"]))
                progress.update(int(trajectory["env_steps"]))

            ensemble_state, reward_metrics = assign_ensemble_rewards(
                trajectories,
                reward_model,
                ensemble_state,
                intrinsic_scale=float(variant.dppo_intrinsic_reward_scale),
                extrinsic_scale=float(variant.dppo_extrinsic_reward_scale),
                gamma_env=float(variant.dppo_gamma_env),
                running_reward_scale=reward_scale,
            )
            batch = make_dppo_batch(
                trajectories,
                gamma_env=float(variant.dppo_gamma_env),
                gae_lambda=float(variant.dppo_gae_lambda),
            )
            update_info = update_dppo_batch(
                policy,
                value_learner,
                batch,
                num_epochs=int(variant.dppo_num_epochs),
                num_minibatches=int(variant.dppo_num_minibatches),
                minibatch_size=(
                    None
                    if int(variant.dppo_minibatch_size) <= 0
                    else int(variant.dppo_minibatch_size)
                ),
                update_actor=(
                    update_round
                    >= int(variant.dppo_critic_warmup_rounds)
                ),
                seed=int(variant.seed) + update_round,
            )
            update_round += 1

            elapsed = max(time.monotonic() - start_time, 1e-8)
            log_data = {
                **{f"training/{key}": value
                   for key, value in update_info.items()},
                **{f"dynamics/{key}": value
                   for key, value in reward_metrics.items()},
                "env_steps": total_env_steps,
                "num_online_samples": total_queries,
                "num_online_trajs": total_trajectories,
                "dppo/update_round": update_round,
                "dppo/rollout_queries": rollout_queries,
                "dppo/wall_clock_seconds": elapsed,
                "dppo/env_steps_per_second": (
                    total_env_steps - session_start_env_steps) / elapsed,
                "episode_return (exploration)": float(
                    np.mean(rollout_returns)),
                "is_success (exploration)": float(
                    np.mean(rollout_successes)),
                "interaction/contact_frequency": float(
                    np.mean(contact_frequencies)),
            }
            wandb_logger.log(log_data, step=total_env_steps)

            if _crossed_interval(
                    previous_env_steps,
                    total_env_steps,
                    int(variant.eval_interval)):
                evaluate_dppo_policy(
                    policy,
                    eval_env,
                    variant,
                    wandb_logger,
                    step=total_env_steps,
                )

            if _crossed_interval(
                    previous_env_steps,
                    total_env_steps,
                    int(variant.checkpoint_interval)):
                save_dppo_checkpoint(
                    variant.outputdir,
                    total_env_steps,
                    policy=policy,
                    value_learner=value_learner,
                    ensemble_state=ensemble_state,
                    reward_scale=reward_scale,
                    counters={
                        "total_env_steps": total_env_steps,
                        "total_queries": total_queries,
                        "total_trajectories": total_trajectories,
                        "update_round": update_round,
                    },
                )


__all__ = (
    "RunningRewardScale",
    "assign_ensemble_rewards",
    "collect_dppo_trajectory",
    "compute_dppo_gae",
    "evaluate_dppo_policy",
    "load_dppo_checkpoint",
    "make_dppo_batch",
    "save_dppo_checkpoint",
    "trajwise_dppo_training_loop",
    "update_dppo_batch",
)
