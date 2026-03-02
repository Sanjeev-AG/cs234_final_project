# This file defines the goal wrapper for the Seaquest environment:
import gymnasium as gym
import torch
import numpy as np
import random

class SeaQWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SeaQWrapper, self).__init__(env)
        self.num_goal_dimension = 3

        # Initializing SeaQuest parameters:
        self.num_divers_collected = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0
        self.num_lives_left = 3 # By default there are 3 lives left.

        self.desired_goal = None
        self.max_divers_to_collect = 6
        self.max_num_surfaced_count = 1
        self.max_num_attackers_to_shoot = 30


        # Define the goal space:
        # Rescuing 6 divers and resurfacing once and killing 10 attackers would amount to 2000 + 200 = 2200 points.
        # So, the goal of HER is to accumulate enough points for learning a policy to maximize the reward.
        self.achieved_goal = self.get_achieved_goal()

    def reset(self, **kwargs):
        self.num_divers_collected = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0

        self.achieved_goal = self.get_achieved_goal()
        self.num_lives_left = 3

        if "seed" in kwargs:
            np.random.seed(kwargs["seed"])
            torch.manual_seed(kwargs["seed"])

        # Sample a goal from the set of goals for the Hindsight replay:
        self.sample_goal()

        return self.env.reset(**kwargs)


    def get_achieved_goal(self):
        return torch.tensor([self.num_attackers_shot, self.num_divers_collected, self.num_surfaced_count])

    def step(self, action):
        next_obs, reward, terminated, truncated, _ = self.env.step(action)

        self.update_objective_values(reward, next_obs)
        reward_her = self.compute_reward()

        return next_obs, reward, terminated, truncated, reward_her

    def compute_reward(self):
        self.achieved_goal = self.get_achieved_goal()
        reward = 0
        for idx, goal in enumerate(self.desired_goal):
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

    def sample_goal(self):
        # This function helps to sample the desired goal.

        num_attackers_to_shoot = random.randint(a=1, b=self.max_num_attackers_to_shoot)
        num_resurfaces = random.randint(a=0, b=self.max_num_surfaced_count)
        num_divers_to_collect = random.randint(a=0, b=6) + 6*num_resurfaces

        self.desired_goal = torch.tensor([num_attackers_to_shoot, num_divers_to_collect, num_resurfaces])

    def set_max_goals(self, average_reward):
        # ToDo: Dynamically set the max goals once the network has achieved them
        if average_reward > 200:
            # Here, the agent has not learnt how to rescue the divers yet and is focused on attacking.
            self.max_num_attackers_to_shoot += 10

        pass


