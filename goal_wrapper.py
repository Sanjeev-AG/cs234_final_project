"""
This file defines the goal wrapper for the Seaquest environment.
"""

import gymnasium as gym
import torch
import numpy as np
import random
from collections import deque

class SeaQWrapper(gym.Wrapper):
    """
    A wrapper for the Seaquest environment that defines a goal space and computes
    rewards based on the achieved goals.

    The goal space is defined as a 2-dimensional vector:
    1. Number of divers collected
    2. Oxygen meter bucket (0: High, 1: Medium, 2: Low)
    """
    def __init__(self, env, config):
        super(SeaQWrapper, self).__init__(env)

        self.config = config
        self.updated_max_goals = False
        self.pos_goal_num_divers = 0
        self.pos_goal_oxygen = 1

        # Dimensions updated based on our new goal definition
        self.num_goal_dimension = 2

        self.max_reward = -10
        self.episode_success = False

        # Initializing SeaQuest parameters:
        self.num_divers_collected = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0
        self.submarine_y_vector = 0
        self.prev_num_attackers_shot = self.num_attackers_shot

        # Internal variables:
        self.curr_num_lives_left = 3
        self.prev_num_lives_left = self.curr_num_lives_left
        self._previous_state_num_divers = 0
        self.current_oxygen_level = 64 # Max oxygen in Seaquest RAM

        self.desired_goal = None
        self.max_divers_to_collect: int = 1

        self.num_divers_history_buffer = deque(maxlen=100)

        self.achieved_goal = self.get_achieved_goal()

    def reset(self, **kwargs):
        self.num_divers_collected = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0
        self.submarine_y_vector = 0
        self.max_reward = -10
        self.prev_num_attackers_shot = self.num_attackers_shot

        self._previous_state_num_divers = 0
        self.current_oxygen_level = 64

        self.curr_num_lives_left = 3
        self.prev_num_lives_left = self.curr_num_lives_left

        self.num_divers_history_buffer.append(self.episode_success)
        self.episode_success = False
        self.update_max_goals()

        if "seed" in kwargs:
            np.random.seed(kwargs["seed"])
            torch.manual_seed(kwargs["seed"])

        # Sample a goal from the set of goals for the Hindsight replay:
        self.sample_goal()

        obs, info = self.env.reset(**kwargs)
        obs = obs.flatten()
        self.achieved_goal = self.get_achieved_goal()
        return obs, info

    def get_oxygen_bucket(self, oxygen_val):
        """
        Discretize oxygen into 3 buckets:
        Max is 64.
        High (>42), Med (21-42), Low (<21)
        We normalize these buckets to [0.0, 0.5, 1.0] for the neural net.
        """
        if oxygen_val > 42:
            return 0.0 # High
        elif oxygen_val > 21:
            return 0.5 # Med
        else:
            return 1.0 # Low

    def get_achieved_goal(self):
        return torch.tensor([self.num_divers_collected, self.get_oxygen_bucket(self.current_oxygen_level)])

    def get_max_reward(self):
        return self.max_reward

    def step(self, action):
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        next_obs = next_obs.flatten()
        self.update_objective_values(reward, next_obs)
        reward_her = self.compute_reward(action)

        # Modify the observation for the model to work on:
        # next_obs = np.concatenate((next_obs, [self.num_attackers_shot, self.num_surfaced_count))

        return next_obs, reward, terminated, truncated, reward_her

    def compute_reward(self, action):
        """
        Computes the reward based on the CURRENT state of the submarine.
        Success is defined as: Having target divers AND target oxygen bucket AND being at the surface.
        """
        self.achieved_goal = self.get_achieved_goal()

        desired_num_divers = self.desired_goal[self.pos_goal_num_divers]
        desired_oxy_bucket = self.desired_goal[self.pos_goal_oxygen]

        # 1. Check Divers
        divers_ok = self.num_divers_collected >= (desired_num_divers * 0.99)

        # 2. Check Oxygen Bucket
        current_oxy_bucket = self.get_oxygen_bucket(self.current_oxygen_level)
        oxy_ok = abs(current_oxy_bucket - desired_oxy_bucket) < 0.1

        # 3. Check Location (Must be near the surface to count as a "resurface success")
        # In RAM 97, values < 25 are generally considered "surface"
        at_surface = self.submarine_y_vector < (25 / self.config.max_state_value)

        # Must satisfy all 3 state conditions to get the 0.0 reward
        if divers_ok and oxy_ok and at_surface and self.num_divers_collected > 0:
            reward = 50
            self.episode_success = True
        else:
            reward = 0

        # Attacker bonus
        bonus = self.config.attackers_weight * (self.num_attackers_shot - self.prev_num_attackers_shot)
        firing_actions = [1] + list(range(10,18))

        if bonus > 0:
            reward += bonus
        elif action in firing_actions:
            # Continuous firing actions are observed towards the right of the screen.
            # This negative reward should prevent firing actions potentially.
            reward -= 0.1


        self.prev_num_attackers_shot = self.num_attackers_shot

        if reward > self.max_reward:
            self.max_reward = reward

        return reward

    def update_objective_values(self, reward, state):
        offset = (self.config.stack_size - 1) * 128 # We get the state of the latest frame.,

        if reward in [20.0, 30.0]:
            self.num_attackers_shot += 1

        self.curr_num_lives_left = state[offset + 59]
        self.current_oxygen_level = state[offset + 102]

        # Track divers normally
        self.num_divers_collected = self.normalize_divers(state[offset + 62])
        self.submarine_y_vector = self._normalize_state_value(state[offset + 97])

        # Track resurface events strictly for logging purposes
        if state[offset + 62] < self._previous_state_num_divers and self.curr_num_lives_left == self.prev_num_lives_left and self.submarine_y_vector < (25 / self.config.max_state_value):
            self.num_surfaced_count += 1

        self._previous_state_num_divers = state[offset + 62]
        self.prev_num_lives_left = self.curr_num_lives_left

    def sample_goal(self):
        """
        Samples a new desired goal.
        Forces the agent to learn to surface with varying amounts of oxygen remaining.
        """
        # Sample divers
        if random.random() > 0.7:
            num_divers_to_collect = self.normalize_divers(random.randint(a=1, b=self.max_divers_to_collect))
        else:
            num_divers_to_collect = self.normalize_divers(self.max_divers_to_collect)

        # Sample oxygen bucket (0.0 = High, 0.5 = Med, 1.0 = Low)
        # We bias toward medium/low to force longer dives
        oxy_bucket = random.choice([0.0, 0.5, 0.5, 1.0, 1.0])

        self.desired_goal = torch.tensor([num_divers_to_collect, oxy_bucket])

    def update_max_goals(self):
        """
        Only increase difficulty if the agent succeeds in > 60% of the last 100 episodes.
        """
        if len(self.num_divers_history_buffer) < 100:
            return  # Not enough data yet

        success_rate = sum(self.num_divers_history_buffer) / len(self.num_divers_history_buffer)

        if success_rate >= 0.60:  # If it's succeeding 60% of the time
            if self.max_divers_to_collect < self.config.max_divers_rescuable:
                self.max_divers_to_collect += 1
                print(f"*** CURRICULUM ADVANCED! Max divers is now {self.max_divers_to_collect} ***")
                # Clear history so we don't immediately advance again next check
                self.num_divers_history_buffer.clear()

    def normalize_divers(self, num_divers):
        return np.round(num_divers / self.config.max_divers_rescuable, 4)

    def _normalize_state_value(self, state_val):
        return np.round(state_val / self.config.max_state_value, 4)
