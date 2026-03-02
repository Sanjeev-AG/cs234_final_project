import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch

class ReplayBuffer(object):
    def __init__(self, state_dim, goal_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((capacity, state_dim), dtype=torch.float32).to(device)
        self.next_state = torch.zeros((capacity, state_dim), dtype=torch.float32).to(device)
        self.action = torch.zeros((capacity, 1), dtype=torch.int64).to(device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float).to(device)
        self.goal = torch.zeros((capacity, goal_dim), dtype=torch.float32).to(device)
        self.done = torch.zeros((capacity, 1), dtype=torch.uint8).to(device)


    def push(self, state, action, reward, next_state, done, goal):
        with torch.no_grad():
            self.state[self.ptr] = torch.as_tensor(state, dtype=torch.float32).to(device)
            self.next_state[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32).to(device)
            self.action[self.ptr] = torch.as_tensor(action, dtype=torch.int64).to(device)
            self.reward[self.ptr] = torch.as_tensor(reward, dtype=torch.float32).to(device)
            self.done[self.ptr] = torch.as_tensor(done, dtype=torch.uint8).to(device)
            self.goal[self.ptr] = torch.as_tensor(goal, dtype=torch.float32).to(device)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _fetch_indices(self, indices):
        state_batch = self.state[indices]
        next_state_batch = self.next_state[indices]
        action_batch = self.action[indices]
        reward_batch = self.reward[indices]
        done_batch = self.done[indices]
        goal_batch = self.goal[indices]

        return state_batch, next_state_batch, action_batch, reward_batch, done_batch, goal_batch


    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return self._fetch_indices(indices)

    def fetch_last_N_samples(self, N):
        """
        Fetch the last N samples from the experience replay buffer
        :param N: last N samples from self.size
        :return:
        """

        indices = np.arange(self.size-N, self.size)
        return self._fetch_indices(indices)

    def push_batch(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, goal_batch):
        # Get the batch size:
        num_samples = state_batch.shape[0]
        for index in range(num_samples):
            self.push(state_batch[index],
                      action_batch[index],
                      reward_batch[index],
                      next_state_batch[index],
                      done_batch[index],
                      goal_batch)




class DQN(nn.Module):
    """
    Class for implementing DQN
    """

    def __init__(self, env, config):
        """
        env: OpenAI gym environment
        hidden_layer_size: Size of the hidden layer int
        n_layers: int Number of layers
        lr: float learning rate
        gamma: float discount factor
        """
        super().__init__()
        self.env = env
        self.lr = config.lr
        self.gamma = config.gamma
        observation_dim = self.env.observation_space.shape[0]
        if hasattr(env, 'num_goal_dimension'):
            observation_dim+=env.num_goal_dimension
        self.network = build_mlp(input_size=observation_dim, output_size=env.action_space.n,
                                 size=config.layer_size,
                                 n_layers=config.n_layers)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.epsilon = config.epsilon

    def forward(self, obs, goal):
        """
        Args:
            obs: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]

        """
        if isinstance(obs, np.ndarray):
            obs = np2torch(obs)
        obs = torch.cat((obs, goal), dim=-1)
        output = self.network.forward(obs.float()).squeeze()
        return output

    def compute_loss(self, obtained_Q, target_Q):
        """
        Compute the loss for the DQN
        :param obtained_Q:
        :param target_Q:
        :return:
        """

        # MSE error of the loss function
        # loss = torch.nn.functional.mse_loss(obtained_Q, target_Q)
        loss = torch.nn.functional.smooth_l1_loss(obtained_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()

    def select_action(self, in_state, goal=None):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Take completely random action

        with torch.no_grad():  # Don't track gradients during action selection
            output = self.forward(in_state, goal)
            return torch.argmax(output).item()


def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    assert isinstance(x, np.ndarray), f"np2torch expected 'np.ndarray' but received '{type(x).__name__}'"
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x
