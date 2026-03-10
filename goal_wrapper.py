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
        self.curriculum_just_advanced = False
        self.pos_goal_num_divers = 0
        self.pos_goal_oxygen = 1

        # Dimensions updated based on our new goal definition
        self.num_goal_dimension = 2
        self.num_extra_dimension = 1

        self.max_reward = -10
        self.episode_success = False

        # Initializing SeaQuest parameters:
        self.num_divers_collected = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0
        self.previous_num_resurfaced_count = 0
        self.submarine_y_vector = 0
        self.prev_num_attackers_shot = self.num_attackers_shot

        # Internal variables:
        self.curr_num_lives_left = 3
        self.prev_num_lives_left = self.curr_num_lives_left
        self._previous_state_num_divers = 0

        self.desired_goal = None
        self.max_divers_to_collect: int = 1
        self._pending_surfaced_reset = False

        self.num_divers_history_buffer = deque(maxlen=100)

        self.achieved_goal = self.get_achieved_goal()

    def reset(self, **kwargs):
        self.num_divers_collected = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0
        self.previous_num_resurfaced_count = 0
        self.submarine_y_vector = 0
        self.max_reward = -10
        self.prev_num_attackers_shot = self.num_attackers_shot
        self._pending_surfaced_reset = False

        self._previous_state_num_divers = 0

        self.curr_num_lives_left = 3
        self.prev_num_lives_left = self.curr_num_lives_left

        if "seed" in kwargs:
            np.random.seed(kwargs["seed"])
            torch.manual_seed(kwargs["seed"])
        else:
            if self.desired_goal[0] == self.max_divers_to_collect:
                self.num_divers_history_buffer.append(self.episode_success)

        # The queue is based on the maximum number of divers collected:
        self.episode_success = False
        self.update_max_goals()


        # Sample a goal from the set of goals for the Hindsight replay:
        self.sample_goal()

        obs, info = self.env.reset(**kwargs)
        obs = obs.flatten()
        self.achieved_goal = self.get_achieved_goal()
        return obs, info

    def get_achieved_goal(self):
        return torch.tensor([self.num_divers_collected, self.num_surfaced_count])

    def get_max_reward(self):
        return self.max_reward

    def step(self, action):
        if self._pending_surfaced_reset:
            self.num_surfaced_count = 0
            self._pending_surfaced_reset = False

        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        next_obs = next_obs.flatten()
        self.update_objective_values(reward, next_obs)
        reward_her = self.compute_reward(action)

        # Modify the observation for the model to work on:
        next_obs = np.concatenate((next_obs, [self.num_surfaced_count]))

        has_resurfaced = self.num_surfaced_count > self.previous_num_resurfaced_count
        self.previous_num_resurfaced_count = self.num_surfaced_count

        return next_obs, reward, terminated, truncated, reward_her, has_resurfaced

    def compute_reward(self, action):
        """
        Computes the reward based on the CURRENT state of the submarine.
        Success is defined as: Having target divers AND target oxygen bucket AND being at the surface.
        """
        self.achieved_goal = self.get_achieved_goal()

        reward = 0

        # Check if there is a resurfacing event:
        with torch.no_grad():
            if torch.equal(self.achieved_goal, self.desired_goal):
                reward = 50
                reward += self.curr_num_lives_left * 20
                self._pending_surfaced_reset = True  # Reset surfaced count on the next step
                self.episode_success = True

        # bonus = self.config.attackers_weight * (self.num_attackers_shot - self.prev_num_attackers_shot)
        #
        # if bonus > 0:
        #     reward += bonus

        self.prev_num_attackers_shot = self.num_attackers_shot

        if reward > self.max_reward:
            self.max_reward = reward

        return reward

    def update_objective_values(self, reward, state):
        offset = (self.config.stack_size - 1) * 128 # We get the state of the latest frame.,

        if reward in [20.0, 30.0]:
            self.num_attackers_shot += 1

        self.curr_num_lives_left = state[offset + 59]

        # Track divers normally
        self.num_divers_collected = self._previous_state_num_divers
        self.submarine_y_vector = state[offset + 97]

        # Track resurface events strictly for logging purposes
        if state[offset + 62] < self._previous_state_num_divers and self.curr_num_lives_left == self.prev_num_lives_left and self.submarine_y_vector in [13,14,15,16,17]:
            self.num_surfaced_count += 1


        self._previous_state_num_divers = state[offset + 62]
        self.prev_num_lives_left = self.curr_num_lives_left

    def sample_goal(self):
        """
        Samples a new desired goal.
        Forces the agent to learn to surface with varying amounts of oxygen remaining.
        """
        # Sample divers
        if random.random() > 0.2:
            num_divers_to_collect = random.randint(a=1, b=self.max_divers_to_collect)
        else:
            num_divers_to_collect = self.max_divers_to_collect

        # if random.random() <= 1:
        desired_num_surfaced_count = 1
        # else:
        #     desired_num_surfaced_count = random.choice([1, 2, 3])

        self.desired_goal = torch.tensor([num_divers_to_collect, desired_num_surfaced_count])

    def update_max_goals(self):
        """
        Only increase difficulty if the agent succeeds in > 60% of the last 100 episodes.
        """
        if len(self.num_divers_history_buffer) < 100:
            return  # Not enough data yet

        success_rate = sum(self.num_divers_history_buffer) / len(self.num_divers_history_buffer)

        print(f"success rate of the desired goal: {success_rate}")

        if success_rate >= 0.25:  # If it's succeeding 40% of the time
            if self.max_divers_to_collect < self.config.max_divers_rescuable:
                self.max_divers_to_collect += 1
                self.curriculum_just_advanced = True
                print(f"*** CURRICULUM ADVANCED! Max divers is now {self.max_divers_to_collect} ***")
                # Clear history so we don't immediately advance again next check
                self.num_divers_history_buffer.clear()