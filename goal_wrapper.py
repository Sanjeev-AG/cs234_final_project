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
    rewards based on the achieved goals. The goal space is defined as a
    3-dimensional vector consisting of:


        1. Number of divers collected
        2. Did the submarine cross the y vector

    The reward is computed based on whether the achieved goal meets the desired
    goal. If the achieved goal meets or exceeds the desired goal in all
    dimensions, the reward is 0. Otherwise, the reward is -1.

    The wrapper also includes functionality to update the desired goals based
    on the agent's performance and to normalize the goals for input to the model.
    """
    def __init__(self, env, config):
        super(SeaQWrapper, self).__init__(env)

        self.config = config
        self.updated_max_goals = False
        self.pos_goal_num_divers = 0
        self.pos_goal_y_vec = 1
        self.num_goal_dimension = 2
        self.extra_state_dimension = 2

        self.max_reward = -10


        # Initializing SeaQuest parameters:
        self.num_divers_collected = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0
        self.submarine_y_vector = 0

        # Internal variables:
        self.curr_num_lives_left = 3 # By default there are 3 lives left.
        self.prev_num_lives_left = self.curr_num_lives_left
        self._previous_state_num_divers = None
        self.desired_goal = None
        self.max_divers_to_collect: int = 1

        # Define the goal space:
        self.achieved_goal = self.get_achieved_goal()

        self.config = config

    def reset(self, **kwargs):
        """
        Resets the environment and initializes the goal and other parameters.

        Args:
            **kwargs: Seed for reproducibility (optional), plus any other
            arguments for the underlying environment's reset function
        
        Returns:
            obs:    The initial observation after resetting the environment.
            info:   A dictionary containing additional information.
        """
        self.num_divers_collected = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0
        self.submarine_y_vector = 0
        self.max_reward = -10

        self._previous_state_num_divers = 0

        self.achieved_goal = self.get_achieved_goal()
        self.curr_num_lives_left = 3
        self.prev_num_lives_left = self.curr_num_lives_left

        if "seed" in kwargs:
            np.random.seed(kwargs["seed"])
            torch.manual_seed(kwargs["seed"])

        # Update the max goal if required:
        self.update_max_goals()

        # Sample a goal from the set of goals for the Hindsight replay:
        self.sample_goal()

        return self.env.reset(**kwargs)


    def get_achieved_goal(self):
        """
        Returns values for the three dimensions of the achieved goal.
        """
        return torch.tensor([self.num_divers_collected, self.submarine_y_vector])

    def get_max_reward(self):
        """
        Returns the maximum reward obtained from the environment.
        :return:
        """
        return self.max_reward

    def step(self, action):
        """
        Executes the given action in the environment and returns:
            next_obs: The next observation after taking the action.
            reward: The reward obtained after taking the action.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode has been truncated.
            reward_her: The reward computed based on the achieved goal and desired goal for HER.
        """
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        if terminated:
            pass
        self.update_objective_values(reward, next_obs)
        reward_her = self.compute_reward()

        # Modify the observation for the model to work on:
        next_obs = np.concatenate((next_obs, [self._normalize_state_value(self.num_attackers_shot), self._normalize_state_value(self.num_surfaced_count)]))

        return next_obs, reward, terminated, truncated, reward_her

    def compute_reward(self):
        """
        Computes the reward based on the achieved goal and desired goal for HER.

        As per HER, the reward is set as 0 if the desired goal is satisfied (in
        all dimensions) and -1 otherwise.
        """
        self.achieved_goal = self.get_achieved_goal()
        desired_goal = self.desired_goal

        desired_num_divers = desired_goal[self.pos_goal_num_divers]
        desired_y_vec = desired_goal[self.pos_goal_y_vec]

        if self.num_divers_collected >= desired_num_divers and abs(self.submarine_y_vector - desired_y_vec) < self.config.allowable_range_y_vec:
            reward = 0
        else:
            reward = -1

        # Adding the number of attackers as a smaller reward
        reward += self.config.attackers_weight * self.num_attackers_shot

        if reward > self.max_reward:
            self.max_reward = reward

        return reward

    def update_objective_values(self, reward, state):
        """
        Updates the values of the three dimensions of the achieved goal based on the reward and state.
        """
        if reward in [20.0, 30.0]:
            # There are pink attackers worth 30 points and other attackers are 20 points worth
            self.num_attackers_shot += 1

        # If we have a resurfaced with 6 divers, then at minimum the number of points seems to be 6
        # For Seaquest, byte 62 provides the num divers collected

        # The max value oxygen meter can take is 64. It can have the same value across frames:
        # Each time it resurfaces, the oxygen meter increases to 64.
        # self.curr_oxygen_level = state[102]
        self.curr_num_lives_left = state[59]
        if state[62] == 0 and self._previous_state_num_divers == 6 and self.curr_num_lives_left == self.prev_num_lives_left:
            # These conditions would mean the submarine has resurfaced with 6 divers.
            self.num_surfaced_count += 1
        self.prev_num_lives_left = self.curr_num_lives_left

        self.num_divers_collected = self.normalize_divers(state[62])
        self.submarine_y_vector = self._normalize_state_value(state[97])

    def sample_goal(self):
        """
        Samples a new desired goal from the set of goals for the Hindsight replay.
        The goal is sampled randomly from the range of possible goals,
        but can be biased towards higher goals to encourage exploration.
        """
        if random.random() < 0.7:
            num_divers_to_collect = self.normalize_divers(random.randint(a=1, b=self.max_divers_to_collect))
            y_vector = self._normalize_state_value(random.randint(a = 13, b=20))
        else:
            num_divers_to_collect = self.normalize_divers(self.max_divers_to_collect)
            y_vector = self._normalize_state_value(13)


        self.desired_goal = torch.tensor([num_divers_to_collect, y_vector])

    def update_max_goals(self):
        """
        Updates the maximum goals for each dimension based on the agent's
        performance in the recent history.

        If the agent has achieved the desired goal in more than 80% of the
        recent episodes, we can increase the maximum goals for that dimension
        to encourage further learning and exploration.
        """

        # Increment by 2 divers to collect if the achieved goal exceeds by 90%
        self.max_divers_to_collect += 1


    def normalize_divers(self, num_divers):
        return round(num_divers / self.config.max_divers_rescuable, 4)

    def _normalize_state_value(self, state_val):
        return round(state_val / self.config.max_state_value, 4)



