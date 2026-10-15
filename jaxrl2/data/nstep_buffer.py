from jaxrl.datasets import ReplayBuffer
import numpy as np
import collections

NstepBatch = collections.namedtuple(
    'NstepBatch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations',
     'n_step_rewards', 'n_step_masks', 'n_step_next_observations'])


class NstepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int,
                 discount: float = 0.99, n_steps: int = 3, *args, **kwargs):
        if capacity < 1:
            raise ValueError('capacity must be positive')
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

        total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.masks.nbytes
                + self.dones_float.nbytes + self.next_observations.nbytes
        )

        total_memory_usage /= 1e9
        print(
            "Memory requirements"
            f"replay buffer {total_memory_usage:.2f}GB"
        )

    def initialize_with_dataset(self, dataset, num_samples=None):
        super().initialize_with_dataset(dataset, num_samples)
        self.insert_index %= self.capacity

    def sample(self, batch_size: int) -> NstepBatch:
        n_steps = self.n_steps
        step_indx = np.random.randint(self.size, size=batch_size)
        observation = self.observations[step_indx]
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
            # next_obs_ = \
            #      {key: stopping_criteria.reshape((-1, )
            #                         + (1,) * len(next_obs_[key].shape[1:])) * next_obs_[key]
            #            + (1 - stopping_criteria.reshape((-1, )
            #                         + (1,) * len(next_obs_[key].shape[1:]))) * obs[current_idx, env_indices, :]
            #       for key, obs in self.next_observations.items()}
            next_obs_idx = stopping_criteria * next_obs_idx + (1 - stopping_criteria) * current_idx

            done = done * stopping_criteria + (1 - stopping_criteria) * self.dones_float[current_idx]
            stopping_criteria = stopping_criteria * stopping_criteria + (1 - stopping_criteria) * (
                    1 - (1 - done) * (1 - (current_idx == (self.insert_index - 1) % self.capacity)))
            # stopping_criteria = stopping_criteria.astype(reward.dtype)

        reward = reward.astype(reward_dtype)
        masks = masks.astype(reward_dtype)
        gamma = gamma.astype(reward_dtype)
        done = done.astype(reward_dtype)
        next_obs_idx = next_obs_idx.astype(next_obs_idx_dtype)
        next_observation = self.next_observations[next_obs_idx].astype(observation.dtype)

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
            next_observations=self.next_observations[step_indx],
            n_step_next_observations=next_observation,
            n_step_rewards=reward,
            n_step_masks=discount.astype(self.masks.dtype),  # V(s) = \sun^{n-1}_{t=0}\gamma^t r_t + \gamma mask V(s')
            # -> if not terminate. mask = \gamma^{n-1}, if terminate mask = 0.0
        )
