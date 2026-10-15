import chex
from jaxtyping import PyTree
from typing import Optional, Dict
import collections

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict as ObsDict
from gymnasium.spaces import Box


@chex.dataclass
class TactileObs:
    image: PyTree
    tactile: Optional[PyTree]


def convert_obs_to_tact_obs(observation: Dict, next_observation: Optional[Dict] = None):
    if 'tactile' in observation:
        tact_obs = TactileObs(
            image=observation['image'],
            tactile=observation['tactile']
        )
        if next_observation is not None:
            tact_next_obs = TactileObs(
                image=next_observation['image'],
                tactile=next_observation['tactile']
            )
            return tact_obs, tact_next_obs
        return tact_obs
    else:
        tact_obs = TactileObs(
            image=observation['image'],
            tactile=None,
        )
        if next_observation is not None:
            tact_next_obs = TactileObs(
                image=next_observation['image'],
                tactile=None
            )
            return tact_obs, tact_next_obs
        return tact_obs


class TactileObservationWrapper(gym.ObservationWrapper):
    """Adapt tactile_envs observations to the learners' uint8/NHWC convention."""

    def __init__(self, env, tactile_size: int):
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        image_space = spaces['image']
        self._unit_image = (
            np.issubdtype(image_space.dtype, np.floating)
            and np.all(image_space.low >= 0)
            and np.all(image_space.high <= 1)
        )
        spaces['image'] = Box(0, 255, shape=image_space.shape, dtype=np.uint8)
        self._channels_first = False
        if 'tactile' in spaces:
            tactile_space = spaces['tactile']
            self._channels_first = (
                len(tactile_space.shape) == 3
                and tactile_space.shape[-2:] == (tactile_size, tactile_size)
            )
            low, high = tactile_space.low, tactile_space.high
            if self._channels_first:
                low, high = np.moveaxis(low, 0, -1), np.moveaxis(high, 0, -1)
            spaces['tactile'] = Box(low=low, high=high, dtype=np.float32)
        self.observation_space = ObsDict(spaces)

    def _image(self, image):
        image = np.asarray(image)
        if self._unit_image:
            image = image * 255.0
        return np.rint(np.clip(image, 0, 255)).astype(np.uint8)

    def observation(self, observation):
        observation = dict(observation)
        observation['image'] = self._image(observation['image'])
        if 'tactile' in observation:
            tactile = np.asarray(observation['tactile'])
            if self._channels_first:
                tactile = np.moveaxis(tactile, 0, -1)
            observation['tactile'] = tactile.astype(np.float32)
        return observation

    def render(self, *args, **kwargs):
        return self._image(self.env.render(*args, **kwargs))

    def close(self):
        try:
            self.env.close()
        finally:
            # Older tactile_envs inherits Env.close without releasing its renderer.
            renderer = getattr(self.env.unwrapped, 'renderer', None)
            if renderer is not None and hasattr(renderer, 'close'):
                renderer.close()


# From https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L229
# and modified for memory efficiency.
class LazyFrames(object):

    def __init__(self, frames, stack_axis=-1):
        self._frames = frames
        self._stack_axis = stack_axis

    def __array__(self, dtype=None, copy=None):
        out = np.concatenate(self._frames, axis=self._stack_axis)
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out


class DictFrameStack(gym.Wrapper):

    def __init__(self, env, num_stack: int, stack_axis=-1, lazy=False):
        super().__init__(env)
        if num_stack < 1:
            raise ValueError('num_stack must be positive')
        self._num_stack = num_stack
        self._stack_axis = stack_axis
        self._lazy = lazy

        self.obs_keys = self.observation_space.keys()
        self._frames = {k: collections.deque([], maxlen=num_stack) for k in self.obs_keys}
        spaces = {}

        for k, v in self.observation_space.items():
            low = np.concatenate([v.low] * num_stack, axis=stack_axis)
            high = np.concatenate([v.high] * num_stack, axis=stack_axis)
            box = Box(low=low,
                      high=high,
                      dtype=v.dtype)

            spaces[k] = box

        self.observation_space = ObsDict(spaces=spaces)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        for frames in self._frames.values():
            frames.clear()
        for _ in range(self._num_stack):
            self.append_obs(obs)
        return self._get_obs(), info

    def append_obs(self, obs):
        for k in self._frames.keys():
            self._frames[k].append(obs[k])

    def step(self, action):
        obs, reward, done, truncate, info = self.env.step(action)
        self.append_obs(obs)
        return self._get_obs(), reward, done, truncate, info

    def _get_obs(self):
        obs = {}
        if self._lazy:
            for key in self.obs_keys:
                assert len(self._frames[key]) == self._num_stack
                obs[key] = LazyFrames(list(self._frames[key]), stack_axis=self._stack_axis)
        else:
            for key in self.obs_keys:
                assert len(self._frames[key]) == self._num_stack
                obs[key] = np.concatenate(list(self._frames[key]), axis=self._stack_axis)
        return obs
