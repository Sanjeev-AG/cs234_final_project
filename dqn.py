import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch


class DQN(nn.Module):
    """
    Class for implementing DQN
    """

    def __init__(self, env, config):
        """
        TODO:
        Create self.network using build_mlp, and create self.optimizer to
        optimize its parameters.
        You should find some values in the config, such as the number of layers,
        the size of the layers, etc.
        The output of the network has dimension 1.
        """
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate
        observation_dim = self.env.observation_space.shape[0]

        #######################################################
        #########   YOUR CODE HERE - 2-8 lines.   #############

        self.network = build_mlp(input_size=observation_dim, output_size=1,
                                 size=self.config.layer_size,
                                 n_layers=self.config.n_layers)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        #######################################################
        #########          END YOUR CODE.          ############

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]

        TODO:
        Run the network forward and then squeeze the result so that it's
        1-dimensional. Put the squeezed result in a variable called "output"
        (which will be returned).

        Note:
        A nn.Module's forward method will be invoked if you
        call it like a function, e.g. self(x) will call self.forward(x).
        When implementing other methods, you should use this instead of
        directly referencing the network (so that the shape is correct).
        """
        #######################################################
        #########   YOUR CODE HERE - 1 lines.     #############

        output = self.network.forward(observations).squeeze()

        #######################################################
        #########          END YOUR CODE.          ############
        assert output.ndim == 1
        return output

