"""
This file contains the implementation of the DQN algorithm, including the replay buffer.
"""

import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch

class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer (stores transitions) for DQN agents.

    A transition is a tuple of (state, action, reward, next_state, done), where:
        state:      the state observed at time t
        action:     the action taken at time t
        reward:     the reward received after taking the action at time t
        next_state: the state observed at time t+1 after taking the action
                    at time t
        done:       a boolean indicating whether the episode ended after
                    taking the action at time t
    """
    def __init__(self, state_dim, capacity, device):
        """
        Args:
            state_dim (int):        The dimension of the state space
            capacity (int):         The maximum number of transitions to store in the buffer
            device (torch.device):  The device to store the tensors on (CPU or GPU)
        """
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((capacity, state_dim), dtype=torch.float32).to(device)
        self.next_state = torch.zeros((capacity, state_dim), dtype=torch.float32).to(device)
        self.action = torch.zeros((capacity, 1), dtype=torch.int64).to(device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float).to(device)
        self.done = torch.zeros((capacity, 1), dtype=torch.uint8).to(device)

    def push(self, state, action, reward, next_state, done):
        """
        Push a transition into the replay buffer.
        """
        self.state[self.ptr] = torch.tensor(state, dtype=torch.float32).to(device)
        self.next_state[self.ptr] = torch.tensor(next_state, dtype=torch.float32).to(device)
        self.action[self.ptr] = torch.tensor(action, dtype=torch.int64).to(device)
        self.reward[self.ptr] = torch.tensor(reward, dtype=torch.float32).to(device)
        self.done[self.ptr] = torch.tensor(done, dtype=torch.uint8).to(device)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Returns:
            state_batch:        torch.Tensor of shape [batch_size, state_dim]
            next_state_batch:   torch.Tensor of shape [batch_size, state_dim]
            action_batch:       torch.Tensor of shape [batch_size, 1]
            reward_batch:       torch.Tensor of shape [batch_size, 1]
            done_batch:         torch.Tensor of shape [batch_size, 1]
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        state_batch = self.state[indices]
        next_state_batch = self.next_state[indices]
        action_batch = self.action[indices]
        reward_batch = self.reward[indices]
        done_batch = self.done[indices]

        return state_batch, next_state_batch, action_batch, reward_batch, done_batch

class DQN(nn.Module):
    """
    Implementation of the DQN algorithm, including the replay buffer.
    """

    def __init__(self, env, config):
        """
        Args:
            env (gym.Env):              OpenAI gym environment
            hidden_layer_size (int):    Size of the hidden layer
            n_layers (int):             Number of layers
            lr (float):                 Learning rate
            gamma (float):              Discount factor
        """
        super().__init__()
        self.env = env
        self.lr = config.lr
        self.gamma = config.gamma
        observation_dim = self.env.observation_space.shape[0]
        self.network = build_mlp(input_size=observation_dim, output_size=env.action_space.n,
                                 size=config.layer_size,
                                 n_layers=config.n_layers)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.epsilon = config.epsilon

    def forward(self, obs):
        """
        Args:
            obs: torch.Tensor of shape [batch_size, dim(observation space)]

        Returns:
            output: torch.Tensor of shape [batch_size]
        """
        if isinstance(obs, np.ndarray):
            obs = np2torch(obs)
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

    def select_action(self, in_state):
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
            output = self.forward(in_state)
            return torch.argmax(output).item()
