import os
import random
import time

import numpy as np
import tqdm
from typing import Optional, Dict, Set
from tensorboardX import SummaryWriter

from jaxrl2.agents import SkillSACLearner
from jaxrl2.data.skill_discovery.buffers import SkillReplayBuffer, NstepSkillReplayBuffer
from jaxrl2.utils.train_utils import make_env, make_humanoid_bench_env as _make_humanoid_bench_env
import wandb
import gymnasium as gym
from jaxrl2.utils.wandb_video import log_new_videos


def make_humanoid_bench_env(
        env_name: str,
        seed: int,
        save_folder: Optional[str] = None,
        add_episode_monitor: bool = True,
        action_repeat: int = 1,
        action_cost: float = 0.0,
        frame_stack: int = 1,
        from_pixels: bool = False,
        pixels_only: bool = True,
        image_size: int = 84,
        sticky: bool = False,
        gray_scale: bool = False,
        flatten: bool = True
):
    return _make_humanoid_bench_env(
        env_name=env_name,
        seed=seed,
        save_folder=save_folder,
        add_episode_monitor=add_episode_monitor,
        action_repeat=action_repeat,
        action_cost=action_cost,
        frame_stack=frame_stack,
        from_pixels=from_pixels,
        pixels_only=pixels_only,
        image_size=image_size,
        sticky=sticky,
        gray_scale=gray_scale,
        flatten=flatten,
    )


def evaluate(agent,
             env: gym.Env,
             num_episodes: int,
             steps_per_skill: int,
             ) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, _ = env.reset()
        skill = agent.sample_skill(observation, temperature=0.0)
        finish = False
        step = 0
        while not finish:
            action = agent.sample_actions(observation, skill, temperature=0.0)
            observation, _, done, truncate, info = env.step(action)
            finish = done or truncate
            step += 1
            if step % steps_per_skill == 0:
                skill = agent.sample_skill(observation, temperature=0.0)
        if 'episode' in info:
            for k in stats.keys():
                stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats


def train(
        project_name: str,
        entity_name: str,
        alg_name: str,
        env_name: str,
        alg_kwargs: Dict,
        env_kwargs: Dict,
        seed: int = 0,
        wandb_log: bool = True,
        log_config: Optional[Dict] = None,
        logs_dir: str = './logs',
        save_video: bool = False,
        replay_buffer_size: int = 1_000_000,
        max_steps: int = 1_000_000,
        use_tqdm: bool = True,
        training_start: int = 0,
        updates_per_step: int = 1,
        steps_per_skill: int = 8,
        batch_size: int = 256,
        log_interval: int = 1_000,
        eval_interval: int = 5_000,
        eval_episodes: int = 5,
        exp_hash: str = '',
        n_steps_returns: int = 4,
):
    run_name = f"{env_name}__{alg_name}__{seed}__{int(time.time())}__{exp_hash}"

    if save_video:
        video_train_folder = os.path.join(logs_dir, 'video', 'train')
        video_eval_folder = os.path.join(logs_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    if 'humanoid_bench' in env_name:
        _, task_name = env_name.split('/')
        env = make_humanoid_bench_env(env_name=task_name, seed=seed, save_folder=video_train_folder, **env_kwargs)
        eval_env = make_humanoid_bench_env(env_name=task_name, seed=seed + 42,
                                         save_folder=video_eval_folder, **env_kwargs)
    else:
        env = make_env(env_name=env_name, seed=seed,
                       save_folder=video_train_folder, **env_kwargs)
        eval_env = make_env(env_name=env_name, seed=seed + 42,
                            save_folder=video_eval_folder, **env_kwargs)

    np.random.seed(seed)
    random.seed(seed)

    logged_train_videos: Set[str] = set()
    logged_eval_videos: Set[str] = set()

    if wandb_log:
        if log_config is None:
            log_config = {'alg': alg_name}
        else:
            log_config.update({'alg': alg_name})
        wandb.init(
            dir=logs_dir,
            project=project_name,
            entity=entity_name,
            sync_tensorboard=True,
            config=log_config,
            name=run_name,
            monitor_gym=False,
            save_code=True)

    summary_writer = SummaryWriter(
        os.path.join(logs_dir, run_name))

    skill_dim = alg_kwargs['skill_dim']
    if alg_name == 'skillsac':
        agent = SkillSACLearner(seed,
                                env.observation_space.sample(),
                                env.action_space.sample(), **alg_kwargs)
    else:
        raise NotImplementedError()
    if n_steps_returns < 0:
        replay_buffer = SkillReplayBuffer(observation_space=env.observation_space,
                                          action_space=env.action_space,
                                          skill_dim=skill_dim,
                                          capacity=replay_buffer_size or max_steps)
    else:
        if 'discount' in alg_kwargs.keys():
            discount = alg_kwargs['discount']
        else:
            discount = 0.99
        replay_buffer = NstepSkillReplayBuffer(observation_space=env.observation_space,
                                               action_space=env.action_space,
                                               skill_dim=skill_dim,
                                               discount=discount,
                                               n_steps=n_steps_returns,
                                               capacity=replay_buffer_size or max_steps)

    eval_returns = []
    observation, _ = env.reset()
    skill = agent.sample_skill(observation)
    for i in tqdm.tqdm(range(1, max_steps + 1),
                       smoothing=0.1,
                       disable=not use_tqdm):

        if i < training_start:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation, skill)
        next_observation, reward, terminate, truncate, info = env.step(action)

        if i % steps_per_skill == 0:
            next_skill = agent.sample_skill(next_observation)
        else:
            next_skill = skill

        if terminate:
            mask = 0.0
        else:
            mask = 1.0

        replay_buffer.insert(observation, action, skill,
                             reward, mask, float(terminate or truncate),
                             next_observation, next_skill)
        observation = next_observation
        skill = next_skill

        if terminate or truncate:
            observation, _ = env.reset()
            # resample a skill upon reset
            skill = agent.sample_skill(observation)
            terminate = False
            truncate = False
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

            if 'is_success' in info:
                summary_writer.add_scalar(f'training/success',
                                          info['is_success'],
                                          info['total']['timesteps'])

            log_new_videos(
                video_train_folder,
                logged_train_videos,
                key_prefix='train_videos',
                step=i,
                enabled=wandb_log and save_video,
            )

        if i >= training_start:
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(batch_size)
                update_info = agent.update(batch)

            if i % log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, eval_episodes, steps_per_skill)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(logs_dir, f'{seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

            log_new_videos(
                video_eval_folder,
                logged_eval_videos,
                key_prefix='eval_videos',
                step=i,
                enabled=wandb_log and save_video,
            )

    env.close()
    eval_env.close()
    summary_writer.close()

    if save_video and wandb_log:
        final_step = max_steps
        log_new_videos(
            video_train_folder,
            logged_train_videos,
            key_prefix='train_videos',
            step=final_step,
            enabled=True,
        )
        log_new_videos(
            video_eval_folder,
            logged_eval_videos,
            key_prefix='eval_videos',
            step=final_step,
            enabled=True,
        )

    if wandb_log:
        wandb.finish()
