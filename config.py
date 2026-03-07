"""
This file contains the configuration for the SeaQuest project, including hyperparameters for the model and training process.
"""

import torch.nn as nn

class SeaQuestConfig:
    def __init__(self):
        # model and training config
        self.batch_size = 256  # number of steps used to compute each policy update
        self.lr = 1e-4
        self.gamma = 0.9  # the discount factor

        # parameters for the policy and baseline models
        self.n_layers = 3
        self.layer_size = [512, 256, 128]
        self.epsilon = 1
        self.epsilon_delay = -3e-7
        self.replay_buffer_size = 2000000
        self.target_update_frequency = 10000

        self.tau_weight = 0.005

        self.max_len_history_buffer = 200
        self.attackers_weight = 0.005 # Additional reward for each attacker shot

        self.max_divers_rescuable = 6
        self.max_state_value = pow(2,8)-1
        self.allowable_range_y_vec = round(5 / self.max_state_value, 4)

        self.goal_update_frequency = 500

