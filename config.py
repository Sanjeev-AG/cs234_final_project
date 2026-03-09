"""
This file contains the configuration for the SeaQuest project, including hyperparameters for the model and training process.
"""

class SeaQuestConfig:
    def __init__(self):
        # model and training config
        self.batch_size = 256  # number of steps used to compute each policy update
        self.lr = 1e-4
        self.gamma = 0.99  # the discount factor

        self.stack_size = 1

        # parameters for the policy and baseline models
        self.n_layers = 3
        self.layer_size = [512, 256, 128]
        self.epsilon = 1
        self.replay_buffer_size = 2000000
        self.target_update_frequency = 10000

        self.attackers_weight = 1 # Additional reward for each attacker shot

        self.priority_base = 3.0
        self.success_multiplier = 5.0

        self.max_divers_rescuable = 6

        self.max_goal_update_rate = 700
