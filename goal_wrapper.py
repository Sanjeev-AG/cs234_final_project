# This file defines the goal wrapper for the Seaquest environment:
import gymnasium as gym
import torch
import numpy as np
import random
from collections import deque

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
        self.max_divers_to_collect = 1
        self.max_num_surfaced_count = 0
        self.max_num_attackers_to_shoot = 5
        self.max_len_history = 100

        self.num_attackers_shot_history = deque([0] * self.max_len_history, maxlen=self.max_len_history)
        self.num_surfaced_count_history = deque([0] * self.max_len_history, maxlen=self.max_len_history)
        self.num_divers_collected_history = deque([0] * self.max_len_history, maxlen=self.max_len_history)


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

        # Update the max goal if required:
        self.update_max_goals()

        # Sample a goal from the set of goals for the Hindsight replay:
        self.sample_goal()

        return self.env.reset(**kwargs)


    def get_achieved_goal(self):
        return torch.tensor([self.num_attackers_shot, self.num_divers_collected, self.num_surfaced_count])

    def update_history(self):
        self.num_attackers_shot_history.append(self.num_attackers_shot >= self.desired_goal[0])
        self.num_divers_collected_history.append(self.num_divers_collected >= self.desired_goal[1])
        self.num_surfaced_count_history.append(self.num_surfaced_count >= self.desired_goal[2])


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

        if float(reward) > 500:
            self.num_surfaced_count += 1

        # State should be an array
        self.num_divers_collected = state[62] + 6*self.num_surfaced_count

        # Currently the num_lives_left is not in goal space.
        # ToDo: Add the num lives left to goal space and see how the policy behaves
        self.num_lives_left = state[59]

    def sample_goal(self):
        # This function helps to sample the desired goal.

        num_attackers_to_shoot = random.randint(a=1, b=self.max_num_attackers_to_shoot)
        num_divers_to_collect = random.randint(a=0, b=self.max_divers_to_collect)
        num_resurfaces = random.randint(a=0, b=num_divers_to_collect//6)

        self.desired_goal = torch.tensor([num_attackers_to_shoot, num_divers_to_collect, num_resurfaces])

    def update_max_goals(self):
        # ToDo: Dynamically set the max goals once the network has achieved them
        if sum(self.num_attackers_shot_history)/len(self.num_attackers_shot_history) > 0.9:
            # Increase by 5 if the achieved goal exceeds by 90%
            self.max_num_attackers_to_shoot += 5
            self.num_attackers_shot_history = deque([0] * self.max_len_history, maxlen=self.max_len_history)

        if sum(self.num_divers_collected_history)/len(self.num_divers_collected_history) > 0.9:
            # Increment by 2 divers to collect if the achieved goal exceeds by 90%
            self.max_divers_to_collect += 2
            self.num_surfaced_count_history = deque([0] * self.max_len_history, maxlen=self.max_len_history)

        if self.max_divers_to_collect > 6:
            self.max_num_surfaced_count = self.max_divers_to_collect // 6




