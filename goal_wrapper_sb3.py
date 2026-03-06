# Goal wrapper compatible with SB3's HerReplayBuffer.
# Wraps SeaQWrapper to expose a Dict observation space with
# "observation", "achieved_goal", and "desired_goal" keys.

import gymnasium as gym
import numpy as np
from goal_wrapper import SeaQWrapper


class SeaQWrapperSB3(SeaQWrapper):
    """Extends SeaQWrapper to provide the Dict obs space that SB3 HER requires."""

    def __init__(self, env, config):
        super().__init__(env, config)

        obs_shape = env.observation_space.shape
        goal_shape = (self.num_goal_dimension,)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.float32),
            "achieved_goal": gym.spaces.Box(low=0, high=1, shape=goal_shape, dtype=np.float32),
            "desired_goal": gym.spaces.Box(low=0, high=1, shape=goal_shape, dtype=np.float32),
        })

    def _make_obs_dict(self, obs):
        return {
            "observation": np.asarray(obs, dtype=np.float32),
            "achieved_goal": self.normalize_goals(self.get_achieved_goal()).numpy(),
            "desired_goal": self.normalize_goals(self.desired_goal).numpy(),
        }

    def reset(self, **kwargs):
        # SB3 passes seed=None; filter it out to avoid torch.manual_seed(None) error
        if kwargs.get("seed") is None:
            kwargs.pop("seed", None)
        obs, info = super().reset(**kwargs)
        return self._make_obs_dict(obs), info

    def step(self, action):
        # Call the grandparent (gym.Wrapper) step to get raw env output,
        # then apply our tracking and reward logic manually to avoid
        # calling compute_reward with the wrong signature.
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.update_objective_values(reward, next_obs)
        her_reward = self._compute_internal_reward()
        obs_dict = self._make_obs_dict(next_obs)
        return obs_dict, reward, terminated, truncated, {"her_reward": her_reward}

    def _compute_internal_reward(self):
        """Internal reward computation (same logic as SeaQWrapper.compute_reward)."""
        return super().compute_reward()

    def compute_reward(self, achieved_goal, desired_goal, info, **kwargs):
        """Called by SB3 HER to recompute rewards for relabeled goals."""
        # achieved_goal and desired_goal are numpy arrays of shape (batch, 3)
        # Return 0 if all goal dimensions are met, -1 otherwise
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal[np.newaxis, :]
            desired_goal = desired_goal[np.newaxis, :]
        success = (achieved_goal >= desired_goal).all(axis=1)
        rewards = np.where(success, 0.0, -1.0)
        return rewards.astype(np.float32)
