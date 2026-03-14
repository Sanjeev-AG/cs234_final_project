"""
Configuration for DQN + HER on Seaquest with two-phase curriculum learning.

Phase 1 (COLLECTION): Learn to collect divers (no resurface requirement).
    Goals: [N, 0] where N = 1..6
    HER relabels with any achieved diver count — very effective.

Phase 2 (RESURFACE): Learn to collect N divers and resurface.
    Goals: [N, 1] where N = 1..6
    HER relabels with ANY future achieved goal, including multi-resurface
    goals like [6, 2] or [6, 3]. These serve as intermediate difficulty
    levels between "easy" (many resurfaces) and "hard" (one resurface).
"""

PHASE_COLLECTION = 0
PHASE_RESURFACE = 1


class SeaQuestConfig:
    def __init__(self):
        # Model architecture
        self.n_layers = 3
        self.layer_size = [512, 256, 128]
        self.stack_size = 1

        # Training
        self.batch_size = 256
        self.lr = 1e-4
        self.gamma = 0.99
        self.epsilon = 1.0
        self.replay_buffer_size = 2_000_000
        self.target_update_frequency = 10_000

        # Goal normalization — divide by ACTUAL max values, not 255
        # max_resurface = 6 because the agent may resurface up to 6 times
        # in an episode (once per diver). HER relabels with [N, 2], [N, 3]
        # etc. as intermediate difficulty goals, so normalization must
        # handle resurface > 1 without values exceeding [0, 1].
        self.goal_max_divers = 6.0
        self.goal_max_resurface = 6.0

        # Curriculum
        self.max_divers_rescuable = 6
        self.curriculum_threshold_by_level = {
            1: 0.35, 2: 0.35, 3: 0.3, 4: 0.25, 5: 0.2, 6:0.15
        }
        self.curriculum_window = 150       # episodes to evaluate over

        self.oxygen_low_threshold = 16
        self.oxygen_low_penalty = -0.5

        # Rewards
        self.goal_reward = 50.0
        self.lives_bonus_weight = 20.0
        self.diver_milestone_bonus = 5.0   # Phase 1 only, per new diver collected
        self.death_penalty = -10.0

        # HER
        self.her_k = 4  # number of HER relabeled goals per transition

        # Priority replay — curriculum-relative (NOT exponential)
        self.priority_frontier = 3.0      # weight for frontier goal transitions
        self.priority_near_frontier = 2.0  # weight for one-below-frontier
        self.priority_old = 1.0           # weight for earlier goals
        self.priority_success_mult = 4.0  # multiplier for successful transitions
        self.priority_resurface_mult = 2.0  # extra multiplier for Phase 2 resurface successes
        self.priority_batch_fraction = 0.6  # fraction of batch that uses priority sampling

        # Epsilon schedule — curriculum-aware
        # Phase 1: initial exploration
        self.phase1_epsilon_start = 1.0
        self.phase1_epsilon_min = 0.1
        self.phase1_initial_decay_steps = 1_000_000  # initial decay period

        # Curriculum advancement bumps (within a phase)
        self.curriculum_epsilon_bump_p1 = 0.3   # bump to this on Phase 1 advance
        self.curriculum_epsilon_bump_p2 = 0.15  # bump to this on Phase 2 advance
        self.curriculum_bump_decay_steps = 300_000  # decay bump back to min over this many steps

        # Phase 2 transition
        self.phase2_epsilon_reset = 0.3
        self.phase2_epsilon_min = 0.05
        self.phase2_epsilon_decay_steps = 500_000

    # RAM byte addresses for Seaquest
    RAM_OXYGEN = 102
    RAM_PLAYER_X = 70
    RAM_PLAYER_Y = 97
    RAM_PLAYER_DIR = 86
    RAM_NUM_LIVES = 59
    RAM_DIVERS_COLLECTED = 62
