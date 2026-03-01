# This file defines the goal wrapper for the Seaquest environment:
import gymnasium as gym
import torch
import numpy as np

class SeaQWrapper(gym.Wrapper):
    def __init__(self, env, num_divers_collected=6, num_attackers_shot=10, num_surfaced_count=1):
        super(SeaQWrapper, self).__init__(env)
        self.original_obs_space = env.observation_space
        self.num_goal_dimension = 3

        # Observation space:
        self.observation_space = gym.spaces.Dict(
            {
                "observation": self.original_obs_space,
                "desired_goal": gym.spaces.Box(low=0, high=torch.inf, shape=(self.num_goal_dimension,), dtype=np.float32),
                "achieved_goal": gym.spaces.Box(low=0, high=torch.inf, shape=(self.num_goal_dimension,), dtype=np.float32),
            }
        )

        # Initializing SeaQuest parameters:
        self.num_divers_collected = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0
        self.num_lives_left = 3 # By default there are 3 lives left.

        # Define the goal space:
        # Rescuing 6 divers and resurfacing once and killing 10 attackers would amount to 2000 + 200 = 2200 points.
        # So, the goal of HER is to accumulate enough points for learning a policy to maximize the reward.
        self.desired_goal = np.array([num_divers_collected, num_divers_collected, num_surfaced_count])
        self.achieved_goal = self._get_achieved_goal()

    def reset(self):
        self.num_divers_collected = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0

        self.achieved_goal = self._get_achieved_goal()
        self.num_lives_left = 3


    def _get_achieved_goal(self):
        return np.array([self.num_divers_collected, self.num_attackers_shot, self.num_surfaced_count])

    def step(self, action):
        next_obs, reward, terminated, truncated, _ = self.env.step(action)

        self.update_objective_values(reward, next_obs)
        reward = self.compute_reward(reward)

        return next_obs, reward, terminated, truncated, self.achieved_goal

    def compute_reward(self, desired_goal):
        self.achieved_goal = self._get_achieved_goal()
        reward = 0
        for idx, goal in enumerate(desired_goal):
            if self.achieved_goal[idx] < goal:
                reward = -1
                break

        # As per HER, the reward is set as 0 if the desired goal is satisfied and -1 if the goal is not achieved
        return reward


    def update_objective_values(self, reward, state):
        if reward in [20.0, 30.0]:
            # There are pink attackers worth 30 points and other attackers are 20 points worth
            self.num_attackers_shot += 1

        # If we have a resurfaced with 6 divers, then at minimum the number of points seems to be 6
        # For Seaquest, byte 62 provides the num divers collected
        # State should be an array
        self.num_divers_collected = state[62]

        if float(reward) > 500:
            self.num_surfaced_count += 1

        # Currently the num_lives_left is not in goal space.
        # ToDo: Add the num lives left to goal space and see how the policy behaves
        self.num_lives_left = state[59]


