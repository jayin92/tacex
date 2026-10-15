import gymnasium as gym
from typing import Union, Optional, Dict
import collections
from jaxrl2.data import NstepBatch
import numpy as np

SkillBatch = collections.namedtuple(
    'SkillBatch',
    ['observations', 'actions', 'skills', 'rewards', 'masks', 'next_observations', 'next_skills'])

NstepSkillBatch = collections.namedtuple(
    'NstepSkillBatch',
    ['observations', 'actions', 'skills', 'rewards', 'masks', 'next_observations', 'next_skills',
     'n_step_next_observations', 'n_nstep_next_skills', 'n_step_rewards', 'n_step_masks',
     ])


class SkillDataset(object):
    def __init__(self,
                 observations: Union[Dict, np.ndarray],
                 actions: np.ndarray,
                 skills: np.ndarray,
                 rewards: np.ndarray,
                 masks: np.ndarray,
                 dones_float: np.ndarray,
                 next_observations: Union[Dict, np.ndarray],
                 next_skills: np.ndarray,
                 size: int):
        if isinstance(observations, Dict):
            self._obs_is_dict = True
        else:
            self._obs_is_dict = False
        self.observations = observations
        self.actions = actions
        self.skills = skills
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.next_skills = next_skills
        self.size = size

    @property
    def obs_is_dict(self):
        return self._obs_is_dict

    def sample(self, batch_size: int) -> SkillBatch:
        indx = np.random.randint(self.size, size=batch_size)
        if self.obs_is_dict:
            observations = {k: v[indx] for k, v in self.observations.items()}
            next_observations = {k: v[indx] for k, v in self.next_observations.items()}
        else:
            observations = self.observations[indx]
            next_observations = self.next_observations[indx]
        return SkillBatch(observations=observations,
                          actions=self.actions[indx],
                          skills=self.skills[indx],
                          rewards=self.rewards[indx],
                          masks=self.masks[indx],
                          next_observations=next_observations,
                          next_skills=self.next_skills[indx],
                          )


class SkillReplayBuffer(SkillDataset):

    def __init__(self,
                 observation_space: Union[gym.spaces.Box, gym.spaces.Dict],
                 action_space: Union[gym.spaces.Discrete, gym.spaces.Box],
                 skill_dim: int,
                 capacity: int):

        if capacity < 1:
            raise ValueError('capacity must be positive')

        if isinstance(observation_space, gym.spaces.Dict):
            observations = {k: np.empty((capacity, *v.shape),
                                        dtype=v.dtype) for k, v in observation_space.items()}
            next_observations = {k: np.empty((capacity, *v.shape),
                                             dtype=v.dtype) for k, v in observation_space.items()}
        else:
            observations = np.empty((capacity, *observation_space.shape),
                                    dtype=observation_space.dtype)
            next_observations = np.empty((capacity, *observation_space.shape),
                                         dtype=observation_space.dtype)

        actions = np.empty((capacity, *action_space.shape),
                           dtype=action_space.dtype)
        skills = np.empty((capacity, skill_dim),
                          dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)

        next_skills = np.empty((capacity, skill_dim),
                               dtype=np.float32)
        super().__init__(observations=observations,
                         actions=actions,
                         skills=skills,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         next_skills=next_skills,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: SkillDataset,
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

        if self.obs_is_dict:
            for k, v in dataset.observations.items():
                self.observations[k][:num_samples] = v[indices]
                self.next_observations[k][:num_samples] = dataset.next_observations[k][indices]
        else:
            self.observations[:num_samples] = dataset.observations[indices]
            self.next_observations[:num_samples] = dataset.next_observations[
                indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.skills[:num_samples] = dataset.skills[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_skills[:num_samples] = dataset.next_skills[indices]

        self.insert_index = num_samples % self.capacity
        self.size = num_samples

    def insert(self,
               observation: Union[np.ndarray, Dict],
               action: np.ndarray,
               skills: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: Union[np.ndarray, Dict],
               next_skills: np.ndarray):
        if self.obs_is_dict:
            for k, v in observation.items():
                self.observations[k][self.insert_index] = v
                self.next_observations[k][self.insert_index] = next_observation[k]
        else:
            self.observations[self.insert_index] = observation
            self.next_observations[self.insert_index] = next_observation
        self.actions[self.insert_index] = action
        self.skills[self.insert_index] = skills
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_skills[self.insert_index] = next_skills

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


class NstepSkillReplayBuffer(SkillReplayBuffer):
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

    def sample(self, batch_size: int) -> NstepSkillBatch:
        n_steps = self.n_steps
        step_indx = np.random.randint(self.size, size=batch_size)
        if self.obs_is_dict:
            observation = {k: v[step_indx] for k, v in self.observations.items()}
            n_obs = {k: v[step_indx] for k, v in self.next_observations.items()}
        else:
            observation = self.observations[step_indx]
            n_obs = self.next_observations[step_indx]
        action = self.actions[step_indx]
        skills = self.skills[step_indx]
        n_skills = self.next_skills[step_indx]

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
        if self.obs_is_dict:
            next_observation = {k: v[next_obs_idx].astype(observation[k].dtype)
                                for k, v in self.next_observations.items()}
        else:
            next_observation = self.next_observations[next_obs_idx]

        next_skills = self.next_skills[next_obs_idx]

        # termination flag
        # Only use dones that are not due to timeouts
        # deactivated by default (timeouts is initialized as an array of False)
        # terminate if done and mask == 0.0, done = terminate and truncate, mask = not terminate
        terminate = done * (1 - masks)
        # add relevant discounting --> 1 - done = \gamma^n if done = False
        discount = (1 - terminate) * gamma

        return NstepSkillBatch(
            observations=observation,
            actions=action,
            skills=skills,
            rewards=self.rewards[step_indx],
            masks=self.masks[step_indx],
            next_observations=n_obs,
            next_skills=n_skills,
            n_step_next_observations=next_observation,
            n_nstep_next_skills=next_skills,
            n_step_rewards=reward,
            n_step_masks=discount.astype(self.masks.dtype),  # V(s) = \sun^{n-1}_{t=0}\gamma^t r_t + \gamma mask V(s')
            # -> if not terminate. mask = \gamma^{n-1}, if terminate mask = 0.0
        )
