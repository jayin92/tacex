"""On-policy DSRL-PPO collection and training utilities.

PPO acts at the same query-level semi-MDP as the existing DSRL-SAC code: one
bounded diffusion-noise vector is chosen, expanded across the pi-zero action
horizon, and executed open loop for up to ``query_freq`` environment steps.
This module is deliberately separate from ``train_utils_sim`` so the replay-
based SAC collection and update path remains unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import Any

import jax
import numpy as np
from tqdm import tqdm

from examples.diffusion_steering.train_utils_sim import (
    apply_residual_delta,
    get_actual_action_dim,
    get_dsrl_action_mode,
    get_pi0_action_horizon,
    get_pi0_noise_dim,
    make_dsrl_components,
    obs_to_agent_input,
    obs_to_img,
    obs_to_pi_zero_input,
    obs_to_tactile,
    perform_control_eval,
)


def compute_gae(
        rewards: np.ndarray,
        values: np.ndarray,
        bootstrap_value: float,
        discounts: np.ndarray,
        masks: np.ndarray,
        gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes query-level GAE with per-transition semi-MDP discounts.

    ``discounts[t]`` is normally ``gamma ** query_freq`` and uses the actual
    number of executed environment steps for a short final query. ``masks`` is
    zero only for a true termination; time-limit and rollout truncations retain
    a mask of one and therefore bootstrap from ``bootstrap_value``.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    discounts = np.asarray(discounts, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)

    if not (
            rewards.ndim == values.ndim == discounts.ndim == masks.ndim == 1
            and len(rewards) == len(values) == len(discounts) == len(masks)):
        raise ValueError(
            "rewards, values, discounts, and masks must be equal-length 1D "
            "arrays.")
    if not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}.")

    advantages = np.empty_like(rewards)
    next_value = np.float32(bootstrap_value)
    next_advantage = np.float32(0.0)
    for index in range(len(rewards) - 1, -1, -1):
        continuation = discounts[index] * masks[index]
        delta = rewards[index] + continuation * next_value - values[index]
        next_advantage = (
            delta + continuation * np.float32(gae_lambda) * next_advantage)
        advantages[index] = next_advantage
        next_value = values[index]

    return advantages, advantages + values


def _reset_env(env: Any, variant: Any) -> Any:
    reset_output = env.reset()
    if "aloha" in variant.env:
        return reset_output[0]
    # LIBERO currently exposes the legacy Gym reset API. Also tolerate a
    # Gymnasium-style (observation, info) pair for future versions.
    if (
            isinstance(reset_output, tuple)
            and len(reset_output) == 2
            and isinstance(reset_output[1], Mapping)):
        return reset_output[0]
    return reset_output


def _step_env(env: Any, action: np.ndarray) -> tuple[Any, float, bool, bool, Mapping]:
    output = env.step(action)
    if len(output) == 5:
        observation, reward, terminated, truncated, info = output
        return observation, reward, bool(terminated), bool(truncated), info
    if len(output) == 4:
        observation, reward, done, info = output
        info = info or {}
        truncated = bool(
            info.get("TimeLimit.truncated", False)
            or info.get("truncated", False))
        return observation, reward, bool(done and not truncated), truncated, info
    raise ValueError(f"Expected env.step() to return 4 or 5 values, got {len(output)}.")


def _has_contact(observation: Mapping[str, Any], variant: Any) -> bool:
    if not bool(getattr(variant, "use_touch", 0)):
        return False
    try:
        tactile = obs_to_tactile(observation, variant)
    except (KeyError, NotImplementedError):
        return False
    return bool(np.any(np.abs(tactile) > 1e-8))


def collect_ppo_traj(
        variant: Any,
        agent: Any,
        env: Any,
        agent_dp: Any,
        *,
        max_env_steps: int | None = None,
) -> dict[str, Any]:
    """Collects one on-policy episode (or a budget-limited truncation)."""
    if get_dsrl_action_mode(variant) != "noise":
        raise ValueError(
            "DSRL-PPO currently supports diffusion-noise steering only; set "
            "--dsrl_action_mode noise.")
    if agent_dp is None:
        raise ValueError("DSRL-PPO collection requires a loaded pi-zero policy.")

    query_frequency = int(variant.query_freq)
    if query_frequency <= 0:
        raise ValueError(f"query_freq must be positive, got {query_frequency}.")
    episode_limit = int(variant.max_timesteps)
    if max_env_steps is not None:
        episode_limit = min(episode_limit, int(max_env_steps))
    if episode_limit <= 0:
        raise ValueError("max_env_steps must leave room for at least one step.")

    action_horizon = get_pi0_action_horizon(agent_dp)
    pi0_noise_dim = get_pi0_noise_dim(agent_dp)
    actual_action_dim = get_actual_action_dim(variant)
    observation = _reset_env(env, variant)

    observations = []
    actions = []
    pre_tanh_actions = []
    old_log_probs = []
    old_values = []
    query_lengths = []
    raw_query_rewards = []
    total_steps = 0
    contact_steps = 0
    terminated = False
    env_truncated = False
    final_reward = 0.0
    success_seen = False

    while total_steps < episode_limit and not (terminated or env_truncated):
        current_image = obs_to_img(observation, variant)
        agent_observation = obs_to_agent_input(
            observation, variant, curr_image=current_image)
        policy_output = agent.sample_actions_with_info(agent_observation)

        learner_action = np.asarray(policy_output["action"], dtype=np.float32)
        pre_tanh_action = np.asarray(
            policy_output["pre_tanh_action"], dtype=np.float32)
        log_prob = np.asarray(policy_output["log_prob"], dtype=np.float32)
        value = np.asarray(policy_output["value"], dtype=np.float32)
        if learner_action.shape[0] != 1 or pre_tanh_action.shape[0] != 1:
            raise ValueError("PPO trajectory collection expects a single observation batch.")

        observations.append(agent_observation)
        actions.append(learner_action[0])
        pre_tanh_actions.append(pre_tanh_action[0])
        old_log_probs.append(float(log_prob.reshape(-1)[0]))
        old_values.append(float(value.reshape(-1)[0]))

        # The learner's one-row noise action is expanded exactly as in the SAC
        # collector before pi-zero maps it to low-level robot actions.
        learner_action, noise, delta_normalized = make_dsrl_components(
            variant,
            learner_action,
            # Noise-only mode consumes the learned action directly; this key
            # is unused and must not advance the PPO policy's sampling RNG.
            jax.random.PRNGKey(0),
            agent.action_chunk_shape,
            action_horizon,
            pi0_noise_dim,
            actual_action_dim,
        )
        if delta_normalized is not None:
            raise AssertionError("Noise-only DSRL-PPO produced a residual action.")
        pi0_observation = obs_to_pi_zero_input(observation, variant)
        low_level_actions = agent_dp.infer(
            pi0_observation, noise=noise)["actions"]
        # With no residual this still applies the same environment-action
        # bounds as the SAC collector.
        low_level_actions = apply_residual_delta(
            variant, low_level_actions, delta_normalized=None)

        steps_this_query = min(query_frequency, episode_limit - total_steps)
        query_reward = 0.0
        executed = 0
        for action_index in range(steps_this_query):
            if _has_contact(observation, variant):
                contact_steps += 1
            observation, reward, terminated, env_truncated, _ = _step_env(
                env, low_level_actions[action_index])
            final_reward = float(reward)
            success_seen = (
                success_seen
                or final_reward == float(variant.env_max_reward))
            query_reward += float(reward)
            total_steps += 1
            executed += 1
            if terminated or env_truncated:
                break

        raw_query_rewards.append(query_reward)
        query_lengths.append(executed)

    # Reaching either the environment horizon or the global training budget is
    # a truncation, so GAE should bootstrap unless the environment terminated.
    budget_truncated = total_steps >= episode_limit and not terminated
    truncated = bool(env_truncated or budget_truncated)
    final_image = obs_to_img(observation, variant)
    final_observation = obs_to_agent_input(
        observation, variant, curr_image=final_image)

    query_count = len(actions)
    is_success = bool(success_seen)
    # Match the DSRL-SAC query-level sparse reward conversion exactly: each
    # unsuccessful decision costs -1 and a successful final decision costs 0.
    rewards = -np.ones((query_count,), dtype=np.float32)
    if is_success:
        rewards[-1] = 0.0

    masks = np.ones((query_count,), dtype=np.float32)
    if terminated:
        masks[-1] = 0.0
    discounts = np.power(
        np.float32(variant.discount),
        np.asarray(query_lengths, dtype=np.float32),
    ).astype(np.float32)
    bootstrap_value = 0.0
    if masks[-1] != 0.0:
        bootstrap_value = float(
            np.asarray(agent.value(final_observation)).reshape(-1)[0])

    episode_return = float(np.sum(raw_query_rewards))
    contact_frequency = contact_steps / max(total_steps, 1)
    print(
        "PPO rollout done: "
        f"return={episode_return}, success={is_success}, "
        f"queries={query_count}, env_steps={total_steps}, "
        f"terminated={terminated}, truncated={truncated}")

    return {
        "observations": observations,
        "final_observation": final_observation,
        "actions": np.asarray(actions, dtype=np.float32),
        "pre_tanh_actions": np.asarray(pre_tanh_actions, dtype=np.float32),
        "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
        "old_values": np.asarray(old_values, dtype=np.float32),
        "bootstrap_value": np.float32(bootstrap_value),
        "rewards": rewards,
        "discounts": discounts,
        "masks": masks,
        "query_lengths": np.asarray(query_lengths, dtype=np.int32),
        "terminated": terminated,
        "truncated": truncated,
        "is_success": is_success,
        "episode_return": episode_return,
        "env_steps": total_steps,
        "contact_steps": contact_steps,
        "contact_frequency": contact_frequency,
    }


def make_ppo_batch(
        trajectories: Sequence[Mapping[str, Any]],
        gae_lambda: float,
        normalize_advantages: bool = True,
) -> dict[str, Any]:
    """Computes per-trajectory GAE, then concatenates an on-policy batch."""
    if not trajectories:
        raise ValueError("At least one trajectory is required for a PPO batch.")

    observation_keys = tuple(trajectories[0]["observations"][0])
    observation_batches: dict[str, list[np.ndarray]] = {
        key: [] for key in observation_keys}
    flat_parts: dict[str, list[np.ndarray]] = {
        "actions": [],
        "pre_tanh_actions": [],
        "old_log_probs": [],
        "old_values": [],
        "advantages": [],
        "returns": [],
    }

    for trajectory in trajectories:
        advantages, returns = compute_gae(
            trajectory["rewards"],
            trajectory["old_values"],
            trajectory["bootstrap_value"],
            trajectory["discounts"],
            trajectory["masks"],
            gae_lambda,
        )
        for key in observation_keys:
            observation_batches[key].append(np.concatenate(
                [observation[key] for observation in trajectory["observations"]],
                axis=0,
            ))
        for key in ("actions", "pre_tanh_actions", "old_log_probs", "old_values"):
            flat_parts[key].append(np.asarray(trajectory[key]))
        flat_parts["advantages"].append(advantages)
        flat_parts["returns"].append(returns)

    batch = {
        "observations": {
            key: np.concatenate(parts, axis=0)
            for key, parts in observation_batches.items()
        },
        **{
            key: np.concatenate(parts, axis=0)
            for key, parts in flat_parts.items()
        },
    }
    if normalize_advantages:
        advantages = batch["advantages"]
        batch["advantages"] = (
            (advantages - advantages.mean())
            / (advantages.std() + np.float32(1e-8))
        ).astype(np.float32)
    return batch


def _crossed_interval(previous: int, current: int, interval: int) -> bool:
    return interval > 0 and previous // interval < current // interval


def _device_scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in metrics.items():
        value = np.asarray(jax.device_get(value))
        if value.size == 1:
            result[key] = float(value.reshape(()))
    return result


def trajwise_ppo_training_loop(
        variant: Any,
        agent: Any,
        env: Any,
        eval_env: Any,
        wandb_logger: Any,
        *,
        perform_control_evals: bool = True,
        agent_dp: Any = None,
) -> None:
    """Runs simulation-only, on-policy DSRL-PPO without a replay buffer."""
    rollout_steps = int(variant.ppo_rollout_steps)
    if rollout_steps <= 0:
        raise ValueError(
            f"ppo_rollout_steps must be positive, got {rollout_steps}.")
    max_env_steps = int(variant.max_steps)
    gae_lambda = float(variant.ppo_gae_lambda)
    normalize_advantages = bool(variant.ppo_normalize_advantages)

    total_env_steps = 0
    total_queries = 0
    total_trajectories = 0
    update_round = 0
    start_time = time.monotonic()
    wandb_logger.log({
        "num_online_samples": 0,
        "num_online_trajs": 0,
        "env_steps": 0,
        "ppo/update_round": 0,
    }, step=0)
    if perform_control_evals:
        print("Performing evaluation for the initial pi-zero checkpoint.")
        perform_control_eval(
            agent, eval_env, 0, variant, wandb_logger, agent_dp)

    with tqdm(total=max_env_steps, initial=0) as progress:
        while total_env_steps < max_env_steps:
            trajectories = []
            rollout_queries = 0
            rollout_env_steps = 0
            rollout_successes = []
            rollout_returns = []
            rollout_contact_frequencies = []
            previous_env_steps = total_env_steps

            while (
                    rollout_queries < rollout_steps
                    and total_env_steps < max_env_steps):
                remaining_steps = max_env_steps - total_env_steps
                trajectory = collect_ppo_traj(
                    variant,
                    agent,
                    env,
                    agent_dp,
                    max_env_steps=remaining_steps,
                )
                trajectories.append(trajectory)
                query_count = len(trajectory["rewards"])
                rollout_queries += query_count
                total_queries += query_count
                rollout_env_steps += trajectory["env_steps"]
                total_env_steps += trajectory["env_steps"]
                total_trajectories += 1
                rollout_successes.append(float(trajectory["is_success"]))
                rollout_returns.append(float(trajectory["episode_return"]))
                rollout_contact_frequencies.append(
                    float(trajectory["contact_frequency"]))
                progress.update(trajectory["env_steps"])

            batch = make_ppo_batch(
                trajectories,
                gae_lambda=gae_lambda,
                normalize_advantages=normalize_advantages,
            )
            update_info = _device_scalar_metrics(agent.update(batch))
            update_round += 1

            elapsed = max(time.monotonic() - start_time, 1e-8)
            log_data = {
                **{f"training/{key}": value
                   for key, value in update_info.items()},
                "num_online_samples": total_queries,
                "num_online_trajs": total_trajectories,
                "env_steps": total_env_steps,
                "ppo/update_round": update_round,
                "ppo/rollout_queries": rollout_queries,
                "ppo/rollout_env_steps": rollout_env_steps,
                "ppo/wall_clock_seconds": elapsed,
                "ppo/env_steps_per_second": total_env_steps / elapsed,
                "episode_return (exploration)": float(np.mean(rollout_returns)),
                "is_success (exploration)": float(np.mean(rollout_successes)),
                "interaction/contact_frequency": float(
                    np.mean(rollout_contact_frequencies)),
            }
            wandb_logger.log(log_data, step=total_env_steps)

            if _crossed_interval(
                    previous_env_steps, total_env_steps, int(variant.eval_interval)):
                if perform_control_evals:
                    perform_control_eval(
                        agent,
                        eval_env,
                        total_env_steps,
                        variant,
                        wandb_logger,
                        agent_dp,
                    )

            checkpoint_interval = int(variant.checkpoint_interval)
            if _crossed_interval(
                    previous_env_steps, total_env_steps, checkpoint_interval):
                agent.save_checkpoint(
                    variant.outputdir,
                    total_env_steps,
                    checkpoint_interval,
                )
