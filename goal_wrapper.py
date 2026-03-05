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

        1. Number of attackers shot
        2. Number of divers collected
        3. Number of times resurfaced

    The reward is computed based on whether the achieved goal meets the desired
    goal. If the achieved goal meets or exceeds the desired goal in all
    dimensions, the reward is 0. Otherwise, the reward is -1.

    The wrapper also includes functionality to update the desired goals based
    on the agent's performance and to normalize the goals for input to the model.
    """
    def __init__(self, env):
        super(SeaQWrapper, self).__init__(env)
        self.num_goal_dimension = 3
        self.updated_max_goals = False

        # Initializing SeaQuest parameters:
        self.num_divers_collected = 0
        self.num_divers_lost = 0
        self._previous_state_num_divers = 0
        self.num_attackers_shot = 0
        self.num_surfaced_count = 0
        self.curr_num_lives_left = 3 # By default there are 3 lives left.

        self.prev_oxygen_level = 0
        self.curr_oxygen_level = 0
        self.prev_num_lives_left = 0
        self.wait_for_oxygen_refill = True

        self.desired_goal = None
        self.max_divers_to_collect = 1
        self.max_num_surfaced_count = 0
        self.max_num_attackers_to_shoot = 2
        self.max_len_history = 100

        self.max_possible_num_attackers_to_shoot = 5000
        self.max_possible_num_divers_to_collect = 600
        self.max_possible_num_surfaced_count = 40

        self.num_attackers_shot_history = deque([0] * self.max_len_history, maxlen=self.max_len_history)
        self.num_surfaced_count_history = deque([0] * self.max_len_history, maxlen=self.max_len_history)
        self.num_divers_collected_history = deque([0] * self.max_len_history, maxlen=self.max_len_history)

        # Define the goal space:
        # Rescuing 6 divers and resurfacing once and killing 10 attackers would amount to 2000 + 200 = 2200 points.
        # So, the goal of HER is to accumulate enough points for learning a policy to maximize the reward.
        self.achieved_goal = self.get_achieved_goal()

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
        self._previous_state_num_divers = 0
        self.prev_oxygen_level = 0
        self.curr_oxygen_level = 0
        self.prev_num_lives_left = 0
        self.wait_for_oxygen_refill = True

        self.achieved_goal = self.get_achieved_goal()
        self.curr_num_lives_left = 3

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
        return torch.tensor([self.num_attackers_shot, self.num_divers_collected, self.num_surfaced_count])

    def update_history(self):
        """
        Updates history to reflect whether the achieved goal meets the desired goal for each dimension.
        """
        self.num_attackers_shot_history.append(self.num_attackers_shot >= self.desired_goal[0])
        self.num_divers_collected_history.append(self.num_divers_collected >= self.desired_goal[1])
        self.num_surfaced_count_history.append(self.num_surfaced_count >= self.desired_goal[2])


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

        self.update_objective_values(reward, next_obs)
        reward_her = self.compute_reward()

        return next_obs, reward, terminated, truncated, reward_her

    def compute_reward(self):
        """
        Computes the reward based on the achieved goal and desired goal for HER.

        As per HER, the reward is set as 0 if the desired goal is satisfied (in
        all dimensions) and -1 otherwise.
        """
        self.achieved_goal = self.get_achieved_goal()
        reward = 0
        for idx, goal in enumerate(self.desired_goal):
            if self.achieved_goal[idx] < goal:
                reward = -1
                break

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
        
        """
        if self.prev_num_lives_left > self.curr_num_lives_left:
            self.wait_for_oxygen_refill = True
        
        if self.curr_oxygen_level > self.prev_oxygen_level and not self.wait_for_oxygen_refill:
            self.num_surfaced_count += 1
            self.wait_for_oxygen_refill = True

        if self.curr_oxygen_level == 64:
            self.wait_for_oxygen_refill = False

        self.prev_oxygen_level = self.curr_oxygen_level
        """

        if state[62] == 0 and self._previous_state_num_divers == 6 and self.curr_num_lives_left == self.prev_num_lives_left:
            # These conditions would mean the submarine has resurfaced with 6 divers.
            self.num_surfaced_count += 1

        self.prev_num_lives_left = self.curr_num_lives_left

        if state[62] > self._previous_state_num_divers:
            self.num_divers_collected += (state[62] - self._previous_state_num_divers)

        self._previous_state_num_divers = state[62]

    def sample_goal(self):
        """
        Samples a new desired goal from the set of goals for the Hindsight replay.
        The goal is sampled randomly from the range of possible goals,
        but can be biased towards higher goals to encourage exploration.
        """
        if random.random() >= 0.7:
            num_attackers_to_shoot = random.randint(a=1, b=self.max_num_attackers_to_shoot)
            num_divers_to_collect = random.randint(a=1, b=self.max_divers_to_collect)
            num_resurfaces = random.randint(a=1, b=self.max_num_surfaced_count)
        else:
            num_attackers_to_shoot = self.max_num_attackers_to_shoot
            num_divers_to_collect = self.max_divers_to_collect
            num_resurfaces = self.max_num_surfaced_count

        self.desired_goal = torch.tensor([num_attackers_to_shoot, num_divers_to_collect, num_resurfaces])

    def update_max_goals(self):
        """
        Updates the maximum goals for each dimension based on the agent's
        performance in the recent history.

        If the agent has achieved the desired goal in more than 80% of the
        recent episodes, we can increase the maximum goals for that dimension
        to encourage further learning and exploration.
        """
        self.updated_max_goals = False
        if sum(self.num_attackers_shot_history)/self.max_len_history > 0.8:
            # Increase by 5 if the achieved goal exceeds by 90%
            self.max_num_attackers_to_shoot += 1
            self.num_attackers_shot_history.clear()
            self.updated_max_goals = True

        if sum(self.num_divers_collected_history)/self.max_len_history > 0.8:
            # Increment by 2 divers to collect if the achieved goal exceeds by 90%
            self.max_divers_to_collect += 1
            self.num_divers_collected_history.clear()
            self.updated_max_goals = True

        if sum(self.num_surfaced_count_history)/self.max_len_history > 0.8:
            if self.max_num_surfaced_count < self.max_divers_to_collect //6:
                self.max_num_surfaced_count += 1
                self.num_surfaced_count_history.clear()
                self.updated_max_goals = True

    def normalize_goals(self, goal):
        """
        Normalizes the goals in each dimension to ensure they are between 0 and
        1. This helps with training stability and convergence.
        
        The normalization is done using logarithmic scaling to handle the wide
        range of possible goal values. We can also try linear scaling with
        reduced possible goals if logarithmic scaling doesn't work well.
        """

        normalized_goal = torch.zeros([len(self.desired_goal)], dtype=torch.float)
        max_possible_goals = [
            self.max_possible_num_attackers_to_shoot,
            self.max_possible_num_divers_to_collect,
            self.max_possible_num_surfaced_count
        ]
        for i in range(len(goal)):
            normalized_goal[i] = np.log(1+goal[i])/np.log(1+max_possible_goals[i])

        return normalized_goal







