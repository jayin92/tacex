import os
import random
import time

import gymnasium
import numpy as np
import tqdm
import jax.numpy as jnp
from typing import Optional, Dict, Tuple, List, Set
from tensorboardX import SummaryWriter
import copy
from importlib import import_module
from inspect import signature

from jaxrl2.agents import TactileDrQLearner, TactileDrQv2Learner, TactileMaxInfoDrQLearner, \
    TactileMaxInfoDrQv2Learner, TactileDrQExplorer
from jaxrl2.data.tactile.buffers import DictReplayBuffer, DictNstepReplayBuffer
from jaxrl2.utils.tactile.utils import DictFrameStack as FrameStack
from jaxrl2.utils.tactile.utils import TactileObservationWrapper
from jaxrl.evaluation import evaluate
import wandb
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from jaxrl import wrappers
from jaxrl2.utils.wandb_video import log_new_videos


def _environment_class(module_name, class_name):
    module_path = f'tactile_envs.envs.{module_name}'
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as error:
        if error.name and (error.name == module_path or module_path.startswith(error.name + '.')):
            raise ImportError(
                f'The selected environment requires {module_path}, which is absent from '
                'the installed tactile_envs. Install a tactile_envs revision that provides '
                f'{module_name}, or choose an environment available in this checkout.'
            ) from error
        raise
    return getattr(module, class_name)


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             save_freq: int = -1,
             add_episode_monitor: bool = True,
             action_repeat: int = 2,
             action_cost: float = 0.0,
             frame_stack: int = 3,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             tactile_size: int = 32,
             sticky: bool = False,
             env_kwargs: Optional[Dict] = None,
             max_episode_steps: int = 1_000,
             return_grasp_flag_as_reward: bool = True,
             is_eval: bool = False,
             n_objects_start: int = 1,
             n_objects_end: int = 4,
             n_objects_steps_to_end: int = 6_000_000,
             ) -> Tuple[gym.Env, List]:
    assert env_name in ['insertion', 'exploration', 'hand_exploration', 'arm_exploration']
    env_kwargs = dict(env_kwargs or {})
    flatten = False
    if env_name == 'insertion':
        InsertionEnv = _environment_class('insertion', 'InsertionEnv')
        if from_pixels:
            if pixels_only:
                state_type = 'vision'
            else:
                state_type = 'vision_and_touch'
        else:
            raise NotImplementedError

        if env_kwargs is None:
            env_kwargs = {}
        if 'initialize_assets' in signature(InsertionEnv).parameters:
            env_kwargs.setdefault('initialize_assets', True)
        env = InsertionEnv(
            state_type=state_type,
            symlog_tactile=env_kwargs.pop('symlog_tactile', True),
            env_id=-1, im_size=image_size, tactile_shape=(tactile_size, tactile_size),
            **env_kwargs,
        )
        logging_keys = env.logging_keys() if hasattr(env, 'logging_keys') else ['is_success']
        env.render_mode = 'rgb_array'
    elif env_name == 'exploration':
        ExplorationEnv = _environment_class('exploration', 'ExplorationEnv')
        LinearScheduler = _environment_class('exploration', 'LinearScheduler')
        if from_pixels:
            if pixels_only:
                state_type = 'vision'
            else:
                state_type = 'vision_and_touch'
        else:
            raise NotImplementedError

        if env_kwargs is None:
            env_kwargs = {}
        if is_eval:
            env_kwargs['num_init_grasp_steps'] = 0
        if return_grasp_flag_as_reward:
            reward_type = 'grasped'
        else:
            reward_type = 'height'
        n_objects = LinearScheduler(
            start=n_objects_start,
            end=n_objects_end,
            steps_to_end=n_objects_steps_to_end,
        )
        env = ExplorationEnv(
            state_type=state_type,
            env_id=-1, im_size=image_size, tactile_shape=(tactile_size, tactile_size),
            initialize_assets=True,
            reward_type=reward_type,
            n_objects=n_objects,
            **env_kwargs,
        )
        logging_keys = env.logging_keys()
        env.render_mode = 'rgb_array'
    elif env_name == 'arm_exploration':
        ArmEnv = _environment_class('arm', 'ArmEnv')
        ArmLinearScheduler = _environment_class('arm', 'LinearScheduler')
        if from_pixels:
            if pixels_only:
                state_type = 'vision'
            else:
                state_type = 'vision_and_touch'
        else:
            raise NotImplementedError

        if env_kwargs is None:
            env_kwargs = {}
        if is_eval:
            env_kwargs['num_init_grasp_steps'] = 0
        if return_grasp_flag_as_reward:
            reward_type = 'grasped'
        else:
            reward_type = 'height'
        n_objects = ArmLinearScheduler(
            start=n_objects_start,
            end=n_objects_end,
            steps_to_end=n_objects_steps_to_end,
        )
        env = ArmEnv(
            state_type=state_type,
            env_id=-1, im_size=image_size, tactile_shape=(tactile_size, tactile_size),
            initialize_assets=True,
            reward_type=reward_type,
            n_objects=n_objects,
            **env_kwargs,
        )
        logging_keys = env.logging_keys()
        env.render_mode = 'rgb_array'
    elif env_name == 'hand_exploration':
        HandExplorationEnv = _environment_class('hand_exploration', 'HandExplorationEnv')
        if from_pixels:
            env = HandExplorationEnv(
                state_type='vision',
                im_size=image_size,
                start_grasped=False,
                env_id=-1,
                return_grasp_flag_as_reward=return_grasp_flag_as_reward,
            )
            logging_keys = ['grasped']
            env.render_mode = 'rgb_array'
            if not pixels_only:
                from tactile_envs.utils.add_tactile import AddTactile
                env = AddTactile(env)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    env = TactileObservationWrapper(env, tactile_size=tactile_size)
    print(env.render_mode)
    # Seed before EpisodeMonitor, whose reset discards keyword arguments.
    env.reset(seed=seed)
    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    env = wrappers.ActionCost(env, action_cost=action_cost)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        if save_freq > 0:
            env = gym.wrappers.RecordVideo(env, save_folder,
                                           episode_trigger=lambda episode_id: episode_id % save_freq == 0)
        else:
            env = gym.wrappers.RecordVideo(env, save_folder)

    if frame_stack > 1:
        env = FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env, logging_keys


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
        save_freq: int = -1,
        save_train_video: bool = False,
        replay_buffer_size: int = 1_000_000,
        max_steps: int = 1_000_000,
        use_tqdm: bool = True,
        training_start: int = 0,
        updates_per_step: int = 1,
        update_freq: int = 1,
        batch_size: int = 256,
        log_interval: int = 1_000,
        eval_interval: int = 5_000,
        eval_episodes: int = 5,
        exp_hash: str = '',
        n_steps_returns: int = -1,
):
    run_name = f"{env_name}__{alg_name}__{seed}__{int(time.time())}__{exp_hash}"

    if save_video:
        if save_train_video:
            video_train_folder = os.path.join(logs_dir, 'video', 'train')
        else:
            video_train_folder = None
        video_eval_folder = os.path.join(logs_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env, logging_keys = make_env(env_name=env_name, seed=seed,
                                 save_folder=video_train_folder,
                                 save_freq=save_freq,
                                 **env_kwargs)
    eval_env, _ = make_env(env_name=env_name, seed=seed + 42,
                           save_folder=video_eval_folder,
                           save_freq=save_freq,
                           is_eval=True, **env_kwargs)

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

    obs_sample = env.observation_space.sample()
    obs_sample = {k: v[jnp.newaxis] for k, v in obs_sample.items()}
    if alg_name == 'tactile_drq':
        agent = TactileDrQLearner(seed,
                                  obs_sample,
                                  env.action_space.sample()[jnp.newaxis], **alg_kwargs
                                  )
    elif alg_name == 'tactile_drqv2':
        agent = TactileDrQv2Learner(seed,
                                    obs_sample,
                                    env.action_space.sample()[jnp.newaxis], **alg_kwargs
                                    )
    elif alg_name == 'tactile_maxinfodrq':
        agent = TactileMaxInfoDrQLearner(seed,
                                         obs_sample,
                                         env.action_space.sample()[jnp.newaxis], **alg_kwargs
                                         )
    elif alg_name == 'tactile_maxinfodrqv2':
        agent = TactileMaxInfoDrQv2Learner(seed,
                                           obs_sample,
                                           env.action_space.sample()[jnp.newaxis], **alg_kwargs
                                           )
    elif alg_name == 'tactile_drq_explorer':
        agent = TactileDrQExplorer(seed,
                                   obs_sample,
                                   env.action_space.sample()[jnp.newaxis], **alg_kwargs
                                   )
    else:
        raise NotImplementedError()
    if n_steps_returns < 0:
        replay_buffer = DictReplayBuffer(observation_space=env.observation_space,
                                         action_space=env.action_space,
                                         capacity=replay_buffer_size or max_steps)
    else:
        if 'discount' in alg_kwargs.keys():
            discount = alg_kwargs['discount']
        else:
            discount = 0.99
        replay_buffer = DictNstepReplayBuffer(observation_space=env.observation_space, action_space=env.action_space,
                                              discount=discount,
                                              n_steps=n_steps_returns,
                                              capacity=replay_buffer_size or max_steps)

    eval_returns = []
    observation, _ = env.reset()
    log_dictionary = {key: {'max': -10 ** 6, 'total': 0} for key in logging_keys}
    for step in tqdm.tqdm(range(1, max_steps + 1),
                          smoothing=0.1,
                          disable=not use_tqdm):

        if step < training_start:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, terminate, truncate, info = env.step(action)

        if terminate:
            mask = 0.0
        else:
            mask = 1.0

        replay_buffer.insert(observation, action, reward, mask, float(terminate or truncate),
                             next_observation)
        observation = next_observation

        for key in logging_keys:
            if key in info:
                log_dictionary[key]['total'] += copy.deepcopy(info[key])
                log_dictionary[key]['max'] = np.maximum(copy.deepcopy(info[key]), log_dictionary[key]['max'])

        if terminate or truncate:
            observation, _ = env.reset()
            terminate = False
            truncate = False
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

            # if 'is_success' in info:
            #    summary_writer.add_scalar(f'training/success',
            #                              info['is_success'],
            #                              info['total']['timesteps'])
            for key in logging_keys:
                summary_writer.add_scalar(f'training/{key}_total',
                                          copy.deepcopy(log_dictionary[key]['total']),
                                          info['total']['timesteps'])
                summary_writer.add_scalar(f'training/{key}_max',
                                          copy.deepcopy(log_dictionary[key]['max']),
                                          info['total']['timesteps'])
            log_dictionary = {key: {'max': -10 ** 6, 'total': 0} for key in logging_keys}

            log_new_videos(
                video_train_folder,
                logged_train_videos,
                key_prefix='train_videos',
                step=step,
                enabled=wandb_log and save_video,
            )

        if step >= training_start:
            update_info = {}
            if step % update_freq == 0:
                # print(f'Training at {step}')
                for _ in range(updates_per_step):
                    batch = replay_buffer.sample(batch_size)
                    update_info = agent.update(batch)

            if step % log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, step)
                summary_writer.flush()

        if step % eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, eval_episodes)

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
                step=step,
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
