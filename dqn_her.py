"""
This file contains the implementation of the DQN algorithm with Hindsight Experience Replay (HER).
"""

import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, np2torch

class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer (stores transitions) for DQN agents
    with Hindsight Experience Replay (HER).

    A transition is a tuple of (state, action, reward, next_state, done), where:
        state:      the state observed at time t
        action:     the action taken at time t
        reward:     the reward received after taking the action at time t
        next_state: the state observed at time t+1 after taking the action
                    at time t
        done:       a boolean indicating whether the episode ended after
                    taking the action at time t
    """
    def __init__(self, state_dim, goal_dim, capacity, device, config, num_k=4):
        """
        Args:
            state_dim (int):        The dimension of the state space
            goal_dim (int):         The dimension of the goal space
            capacity (int):         The maximum number of transitions to store in the buffer
            device (torch.device):  The device to store the tensors on (CPU or GPU)
            config:                 Configuration object (used for gamma in reward shaping)
            num_k (int):            The number of HER samples to generate for each transition (default 4)
        """
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.num_k = num_k
        self.config = config

        self.state = torch.zeros((capacity, state_dim), dtype=torch.float32).to(device)
        self.next_state = torch.zeros((capacity, state_dim), dtype=torch.float32).to(device)
        self.action = torch.zeros((capacity, 1), dtype=torch.int64).to(device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float).to(device)
        self.goal = torch.zeros((capacity, goal_dim), dtype=torch.float32).to(device)
        self.done = torch.zeros((capacity, 1), dtype=torch.uint8).to(device)


    def push(self, state, action, reward, next_state, done, goal):
        """
        Push a transition into the replay buffer.
        """

        with torch.no_grad():
            self.state[self.ptr] = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            self.next_state[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
            self.action[self.ptr] = torch.as_tensor(action, dtype=torch.int64).to(self.device)
            self.reward[self.ptr] = torch.as_tensor(reward, dtype=torch.float32).to(self.device)
            self.done[self.ptr] = torch.as_tensor(done, dtype=torch.uint8).to(self.device)
            self.goal[self.ptr] = torch.as_tensor(goal, dtype=torch.float32).to(self.device)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _fetch_indices(self, indices):
        """
        Fetches the transitions corresponding to the given indices from the replay buffer.

        Args:
            indices (np.ndarray): An array of indices to fetch from the replay buffer

        Returns:
            state_batch:        torch.Tensor of shape [batch_size, state_dim]
            next_state_batch:   torch.Tensor of shape [batch_size, state_dim]
            action_batch:       torch.Tensor of shape [batch_size, 1]
            reward_batch:       torch.Tensor of shape [batch_size, 1]
            done_batch:         torch.Tensor of shape [batch_size, 1]
        """
        state_batch = self.state[indices]
        next_state_batch = self.next_state[indices]
        action_batch = self.action[indices]
        reward_batch = self.reward[indices]
        done_batch = self.done[indices]
        goal_batch = self.goal[indices]

        return state_batch, next_state_batch, action_batch, reward_batch, done_batch, goal_batch


    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Returns:
            Transitions corresponding to the sampled indices
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return self._fetch_indices(indices)

    def fetch_last_N_samples(self, N):
        """
        Fetches the last N transitions from the replay buffer.

        Args:
            N (int): The number of most recent transitions to fetch

        Returns:
            Transitions corresponding to the last N indices
        """

        indices = np.arange(self.ptr - N, self.ptr) % self.capacity
        return self._fetch_indices(indices)

    def _sample_indices(self, start_idx, stop_idx, success_idx, backup_idx):
        if len(success_idx) > 0:
            valid_values = [x for x in success_idx if x > start_idx]
            if len(valid_values) > 4:
                choices = np.random.choice(valid_values, size=4, replace=False)
            else:
                choices = np.random.choice(valid_values, size=len(valid_values), replace=False)

        else:
            valid_values = [x for x in backup_idx if x > start_idx]
            choices = np.random.choice(valid_values, size=1, replace=False)

        return choices


    def push_batch(self, state_batch, next_state_batch, action_batch, original_rewards_batch, done_batch,
                   obtained_goals, success_idx, backup_idx):
        """
        Push a batch of transitions into the replay buffer, and also generate
        HER samples for each transition in the batch.
        success_idx: Indices where the reward_her was set to 1
        """
        num_samples = state_batch.shape[0]

        for index in range(num_samples - 2):

            # --- EXTRACT COMBAT SHAPING REWARD ---
            # We look at the original reward the agent got on this step.
            # Example 1: Agent fired blindly and missed. original_reward = -1.05
            # Example 2: Agent hit an enemy. original_reward = -0.9
            original_reward = original_rewards_batch[index].item()

            # If the original_reward is greater than -0.5, the agent must have
            # achieved the original goal (base reward 0.0).
            # Otherwise, the base reward was -1.0.
            if original_reward > 45:
                base_original = 50
            else:
                base_original = 0

            combat_shaping = original_reward - base_original
            # ---------------------------------------

            # Sample future states to use as fake HER goals
            goal_sample_indices = self._sample_indices(index, num_samples-1, success_idx, backup_idx)

            for goal_idx in goal_sample_indices:
                # 1. Extract the conditions of the future state we are pretending was the goal
                desired_num_divers = obtained_goals[goal_idx][0]
                num_resurfaced_count = obtained_goals[goal_idx][1]

                # 2. Extract the conditions of the step we are evaluating
                obtained_num_divers = obtained_goals[index + 1][0]
                obtained_num_resurfaced_count = obtained_goals[index + 1][1]

                # If it achieved the divers, oxygen, AND depth of the future state, it's a success!
                if desired_num_divers == obtained_num_divers and num_resurfaced_count == obtained_num_resurfaced_count:
                    if num_resurfaced_count > 0:
                        new_base_reward = 50 # For success indices
                    else:
                        new_base_reward = 1 # For backup indices
                else:
                    new_base_reward = 0

                # Re-apply the combat shaping
                final_her_reward = new_base_reward + combat_shaping

                self.push(state_batch[index],
                          action_batch[index],
                          final_her_reward,
                          next_state_batch[index],
                          done_batch[index],
                          obtained_goals[goal_idx])

    def update_last_index_reward(self):
        """
        Updates the reward of the last transition in the replay buffer to 0.
        """
        last_ptr = (self.ptr-1) % self.capacity
        self.reward[last_ptr] = 0


class DQN(nn.Module):
    """
        Implementation of the DQN algorithm, including the replay buffer.
    """

    def __init__(self, env, config):
        """
        Args:
            env (gym.Env):              OpenAI gym environment
            hidden_layer_size list(int):    Size of the hidden layer
            n_layers (int):             Number of layers
            lr (float):                 Learning rate
            gamma (float):              Discount factor
        """
        super().__init__()
        self.env = env
        self.lr = config.lr
        self.gamma = config.gamma
        if config.stack_size > 0:
            observation_dim = self.env.observation_space.shape[-1] * config.stack_size
        else:
            observation_dim = self.env.observation_space.shape[0]
        if hasattr(env, 'num_goal_dimension'):
            observation_dim+=env.num_goal_dimension
        if hasattr(env, 'num_extra_dimension'):
            observation_dim+=env.num_extra_dimension

        self.network = build_mlp(input_size=observation_dim, output_size=env.action_space.n,
                                 size=config.layer_size,
                                 n_layers=config.n_layers)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.epsilon = config.epsilon

    def forward(self, obs, goal):
        """
        Args:
            obs: torch.Tensor of shape [batch_size, dim(observation space)]

        Returns:
            output: torch.Tensor of shape [batch_size]
        """

        if isinstance(obs, np.ndarray):
            obs = np2torch(obs)
        goal = goal.to(obs.device).float()
        obs = torch.cat((obs, goal), dim=-1)
        output = self.network.forward(obs.float()).squeeze()
        return output

    def compute_loss(self, obtained_Q, target_Q):
        """
        Computes the loss for the DQN algorithm.

        Args:
            obtained_Q: torch.Tensor of shape [batch_size, 1]
                        Q-values obtained from the current network for the batch of transitions
            target_Q:   torch.Tensor of shape [batch_size, 1]
                        Target Q-values computed using the target network
        """

        # MSE error of the loss function
        # loss = torch.nn.functional.mse_loss(obtained_Q, target_Q)
        loss = torch.nn.functional.smooth_l1_loss(obtained_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()

    def select_action(self, in_state, goal=None):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            in_state:   torch.Tensor of shape [dim(observation space)]
                        Current state of the environment
        Returns:
            action:     Action selected by the epsilon-greedy policy
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Take completely random action

        with torch.no_grad():  # Don't track gradients during action selection
            output = self.forward(in_state, goal)
            return torch.argmax(output).item()
