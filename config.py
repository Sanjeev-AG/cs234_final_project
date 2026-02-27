import torch.nn as nn


class SeqQuestConfig:
    def __init__(self):
        # model and training config
        self.batch_size = 128  # number of steps used to compute each policy update
        self.lr = 1e-4
        self.gamma = 0.99  # the discount factor

        # parameters for the policy and baseline models
        self.n_layers = 2
        self.layer_size = 256
        self.epsilon = 1
        self.epsilon_delay = -3e-7
        self.replay_buffer_size = 500000
        self.target_update_frequency = 10000

        self.tau_weight = 0.005
