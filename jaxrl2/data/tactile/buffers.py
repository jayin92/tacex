import gymnasium as gym
from typing import Union, Optional, Dict
from jaxrl.datasets import Batch
from jaxrl2.data import NstepBatch
import numpy as np


class DictDataset(object):
    def __init__(self, observations: Dict, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: Dict,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        observations = {k: v[indx] for k, v in self.observations.items()}
        next_observations = {k: v[indx] for k, v in self.next_observations.items()}
        return Batch(observations=observations,
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=next_observations)


class DictReplayBuffer(DictDataset):

    def __init__(self, observation_space: gym.spaces.Dict,
                 action_space: Union[gym.spaces.Discrete,
                 gym.spaces.Box], capacity: int):

        if capacity < 1:
            raise ValueError('capacity must be positive')

        observations = {k: np.empty((capacity, *v.shape),
                                    dtype=v.dtype) for k, v in observation_space.items()}
        next_observations = {k: np.empty((capacity, *v.shape),
                                         dtype=v.dtype) for k, v in observation_space.items()}
        actions = np.empty((capacity, *action_space.shape),
                           dtype=action_space.dtype)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)

        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: DictDataset,
                                num_samples: Optional[int]):
        assert self.size == 0, 'Can only initialize an empty replay buffer.'

        dataset_size = dataset.size

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)
        for k, v in dataset.observations.items():
            self.observations[k][:num_samples] = v[indices]
            self.next_observations[k][:num_samples] = dataset.next_observations[k][indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]

        self.insert_index = num_samples % self.capacity
        self.size = num_samples

    def insert(self, observation: Dict, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: Dict):
        for k, v in observation.items():
            self.observations[k][self.insert_index] = v
            self.next_observations[k][self.insert_index] = next_observation[k]
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


class DictNstepReplayBuffer(DictReplayBuffer):
    def __init__(self, capacity: int,
                 discount: float = 0.99, n_steps: int = 3, *args, **kwargs):
        super().__init__(capacity=capacity, *args, **kwargs)
        self.discount = discount
        if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
            raise ValueError('n_steps must be a positive integer')
        self.n_steps = n_steps
        # override these elements to be zeros for nstep operation
        del self.rewards, self.masks, self.dones_float
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.masks = np.zeros((capacity,), dtype=np.float32)
        self.dones_float = np.zeros((capacity,), dtype=np.float32)

    def sample(self, batch_size: int) -> NstepBatch:
        n_steps = self.n_steps
        step_indx = np.random.randint(self.size, size=batch_size)
        observation = {k: v[step_indx] for k, v in self.observations.items()}
        action = self.actions[step_indx]

        reward = self.rewards[step_indx]
        # not terminate flag
        masks = self.masks[step_indx]

        # terminate or truncate flag
        done = self.dones_float[step_indx]

        gamma = np.ones_like(done, dtype=reward.dtype)
        # can only sample till pos - 1, after that the data is either empty or from another trajectory
        stopping_criteria = 1 - (1 - done) * (1 - (step_indx == (self.insert_index - 1) % self.capacity))
        stopping_criteria = stopping_criteria.astype(reward.dtype)
        next_obs_idx = step_indx
        reward_dtype = reward.dtype
        next_obs_idx_dtype = next_obs_idx.dtype
        next_obs_idx = next_obs_idx.astype(reward_dtype)
        for idx in range(1, n_steps):
            current_idx = (step_indx + idx) % self.capacity

            reward = reward * stopping_criteria + (1 - stopping_criteria) * (reward + self.discount
                                                                             * gamma * self.rewards[
                                                                                 current_idx])
            masks = masks * stopping_criteria + (1 - stopping_criteria) * self.masks[current_idx]
            gamma = gamma * stopping_criteria + (1 - stopping_criteria) * gamma * self.discount
            next_obs_idx = stopping_criteria * next_obs_idx + (1 - stopping_criteria) * current_idx

            done = done * stopping_criteria + (1 - stopping_criteria) * self.dones_float[current_idx]
            stopping_criteria = stopping_criteria * stopping_criteria + (1 - stopping_criteria) * (
                    1 - (1 - done) * (1 - (current_idx == (self.insert_index - 1) % self.capacity)))

        reward = reward.astype(reward_dtype)
        masks = masks.astype(reward_dtype)
        gamma = gamma.astype(reward_dtype)
        done = done.astype(reward_dtype)
        next_obs_idx = next_obs_idx.astype(next_obs_idx_dtype)
        next_observation = {k: v[next_obs_idx].astype(observation[k].dtype) for k, v in self.next_observations.items()}

        # termination flag
        # Only use dones that are not due to timeouts
        # deactivated by default (timeouts is initialized as an array of False)
        # terminate if done and mask == 0.0, done = terminate and truncate, mask = not terminate
        terminate = done * (1 - masks)
        # add relevant discounting --> 1 - done = \gamma^n if done = False
        discount = (1 - terminate) * gamma
        return NstepBatch(
            observations=observation,
            actions=action,
            rewards=self.rewards[step_indx],
            masks=self.masks[step_indx],
            next_observations={k: v[step_indx] for k, v in self.next_observations.items()},
            n_step_next_observations=next_observation,
            n_step_rewards=reward,
            n_step_masks=discount.astype(self.masks.dtype),  # V(s) = \sun^{n-1}_{t=0}\gamma^t r_t + \gamma mask V(s')
            # -> if not terminate. mask = \gamma^{n-1}, if terminate mask = 0.0
        )
