"""
Goal-conditioned wrapper for Seaquest with two-phase curriculum learning.

Diver tracking models Seaquest's actual mechanics:
- Resurface with 6 divers: deposit all 6. deposited += 6, held = 0.
- Resurface with < 6 divers: lose 1 held diver. held -= 1. No deposit.
- Death: lose 1 held diver. held -= 1. No deposit.

Resurface-with-<6 and death are identical — both just lose 1 held diver.

Achieved goal = [deposited_divers + current_dive_divers, resurface_count]
This gives a realistic total that can decrease (non-monotonic), which HER handles
naturally via the standard "future" relabeling strategy.

Phase 1 (COLLECTION): Goals are [N, 0]. Agent learns to collect N divers total.
Phase 2 (RESURFACE): Goals are [N, 1]. Agent learns to collect and resurface.
    HER relabels with ANY future achieved goal including multi-resurface targets
    like [4, 2] or [6, 3], providing intermediate difficulty levels.
"""

import gymnasium as gym
import torch
import numpy as np
import random
from collections import deque
from config import PHASE_COLLECTION, PHASE_RESURFACE, SeaQuestConfig


class SeaQWrapper(gym.Wrapper):
    def __init__(self, env, config: SeaQuestConfig):
        super().__init__(env)

        self.config = config
        self.num_goal_dimension = 2
        self.num_extra_dimension = 1  # surfaced_count appended to obs

        # Phase and curriculum state
        self.current_phase = PHASE_COLLECTION
        self.max_divers_to_collect = 1

        # Episode tracking
        self._reset_episode_state()

        # Curriculum tracking — separate buffers for each phase
        self.success_history = deque(maxlen=config.curriculum_window)

        self.desired_goal = torch.tensor([1, 0])
        self.achieved_goal = torch.tensor([0, 0])

    def _reset_episode_state(self):
        """Reset all per-episode tracking variables."""
        # Diver tracking — deposit-based accounting
        self.deposited_divers = 0              # Net divers banked across resurfaces (can decrease)
        self.current_dive_divers = 0           # Divers held in submarine right now (from RAM)
        self._prev_ram_divers = 0              # Previous frame's RAM byte 62
        self._milestone_divers_rewarded = 0    # Tracks last rewarded diver count

        # Resurface tracking
        self.num_surfaced_count = 0

        # Submarine state
        self.submarine_y = 0
        self.curr_lives = 3
        self.prev_lives = 3
        self.current_oxygen = 64
        self._oxygen_penalty_given = False

        # Goal tracking
        self._prev_achieved = torch.tensor([0, 0])
        self.episode_success = False
        self.max_reward = -10

        self.peak_divers = 0
        self.deaths_from_oxygen = 0
        self.deaths_from_enemy = 0
        self._died_this_step = False

    def reset(self, **kwargs):
        is_seed_reset = "seed" in kwargs

        if not is_seed_reset:
            # Track success for curriculum (only for frontier goals)
            if self.desired_goal[0] == self.max_divers_to_collect:
                self.success_history.append(self.episode_success)

        self._reset_episode_state()
        self._maybe_advance_curriculum()
        self.sample_goal()

        obs, info = self.env.reset(**kwargs)
        obs = obs.flatten()
        self.achieved_goal = self._compute_achieved_goal()
        self._prev_achieved = self.achieved_goal.clone()
        return obs, info

    def _compute_achieved_goal(self):
        total = self.deposited_divers + self.current_dive_divers
        if self.current_phase == PHASE_COLLECTION:
            return torch.tensor([total, 0])
        else:
            return torch.tensor([total, self.num_surfaced_count])

    def get_achieved_goal(self):
        return self.achieved_goal

    def step(self, action):
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        next_obs = next_obs.flatten()

        self._update_state_from_ram(next_obs)

        self._prev_achieved = self.achieved_goal.clone()
        self.achieved_goal = self._compute_achieved_goal()

        reward_her = self._compute_reward()

        # Append surfaced_count to observation
        next_obs = np.concatenate((next_obs, [self.num_surfaced_count]))

        done = terminated or truncated
        return next_obs, reward, terminated, truncated, reward_her, done

    def _update_state_from_ram(self, state):
        """Extract game state from RAM bytes with diver tracking.

        Seaquest diver mechanics:
        - Resurface with 6 divers: deposit all 6 (only time depositing happens)
        - Resurface with < 6 divers: lose 1 held diver (no deposit)
        - Death: lose 1 held diver (identical to resurface with < 6)
        """
        cfg = self.config
        offset = (cfg.stack_size - 1) * 128

        current_ram_divers = state[offset + cfg.RAM_DIVERS_COLLECTED]
        self.curr_lives = state[offset + cfg.RAM_NUM_LIVES]
        self.submarine_y = state[offset + cfg.RAM_PLAYER_Y]
        self.current_oxygen = state[offset + cfg.RAM_OXYGEN]

        # Detect new diver collection (RAM diver count increased)
        if current_ram_divers > self._prev_ram_divers:
            new_divers = current_ram_divers - self._prev_ram_divers
            self.current_dive_divers += new_divers

        # Detect resurface: diver count dropped, lives unchanged, near surface
        elif (current_ram_divers < self._prev_ram_divers
              and self.curr_lives == self.prev_lives
              and 13 <= self.submarine_y <= 17):
            # Deposit divers with penalty logic
            if self.current_dive_divers >= 6:
                self.deposited_divers += self.current_dive_divers  # Deposit all
                self.current_dive_divers = 0
            else:
                self.current_dive_divers = max(0, self.current_dive_divers - 1)  # Lose 1
            self.num_surfaced_count += 1

        # Detect death: lives decreased
        self._died_this_step = False
        if self.curr_lives < self.prev_lives:
            self._died_this_step = True
            self.current_dive_divers = max(0, self.current_dive_divers - 1)  # Lose 1 held diver
            if self.current_oxygen <= 0:
                self.deaths_from_oxygen += 1
            else:
                self.deaths_from_enemy += 1

        self.peak_divers = max(self.peak_divers, self.deposited_divers + self.current_dive_divers)

        self._prev_ram_divers = current_ram_divers
        self.prev_lives = self.curr_lives

    def _compute_reward(self):
        """
        Compute goal-conditioned reward. Only fires on STATE TRANSITIONS
        (achieved changes from not-matching to matching the desired goal).

        Phase 1 also includes a small per-diver milestone bonus.
        This is safe from reward hacking because collecting divers IS the objective.
        """
        reward = 0.0

        if self._died_this_step:
            reward += self.config.death_penalty

        # Transition-based goal achievement: only reward when achieved JUST matched desired
        prev_match = torch.equal(self._prev_achieved, self.desired_goal)
        curr_match = torch.equal(self.achieved_goal, self.desired_goal)

        if curr_match and not prev_match and not self.episode_success:
            reward += self.config.goal_reward
            reward += self.curr_lives * self.config.lives_bonus_weight
            self.episode_success = True

        # Phase 1 only: per-diver milestone bonus (one-time per diver, up to goal count)
        if self.current_phase == PHASE_COLLECTION:
            goal_divers = self.desired_goal[0].item()
            total = self.deposited_divers + self.current_dive_divers
            if (total > self._milestone_divers_rewarded
                    and self._milestone_divers_rewarded < goal_divers):
                divers_to_reward = min(total, goal_divers)
                new_milestones = divers_to_reward - self._milestone_divers_rewarded
                for i in range(self._milestone_divers_rewarded + 1, divers_to_reward + 1):
                    reward += i * self.config.diver_milestone_bonus
                self._milestone_divers_rewarded = divers_to_reward


        if self.current_oxygen < self.config.oxygen_low_threshold and not self._oxygen_penalty_given:
            reward += self.config.oxygen_low_penalty
            self._oxygen_penalty_given = True

        if reward > self.max_reward:
            self.max_reward = reward

        return reward

    def sample_goal(self):
        """Sample a goal for the current phase and curriculum level."""
        # 70% frontier (hardest current goal), 30% easier goals for maintenance
        if random.random() < 0.7 or self.max_divers_to_collect == 1:
            num_divers = self.max_divers_to_collect
        else:
            num_divers = random.randint(1, self.max_divers_to_collect - 1)

        if self.current_phase == PHASE_COLLECTION:
            self.desired_goal = torch.tensor([num_divers, 0])
        else:
            self.desired_goal = torch.tensor([num_divers, 1])

    def _maybe_advance_curriculum(self):
        """Advance curriculum if success rate exceeds threshold."""
        if len(self.success_history) < self.config.curriculum_window:
            return

        success_rate = sum(self.success_history) / len(self.success_history)
        threshold = self.config.curriculum_threshold_by_level.get(self.max_divers_to_collect, 0.15)
        print(f"  [Curriculum] Phase={'COLLECT' if self.current_phase == PHASE_COLLECTION else 'RESURFACE'}, "
              f"max_divers={self.max_divers_to_collect}, success_rate={success_rate:.2%}")

        if success_rate >= threshold:
            if self.max_divers_to_collect < self.config.max_divers_rescuable:
                self.max_divers_to_collect += 1
                self.success_history.clear()
                print(f"  *** CURRICULUM ADVANCED: max_divers={self.max_divers_to_collect} ***")
            elif self.current_phase == PHASE_COLLECTION:
                # Phase 1 complete — transition to Phase 2
                self.current_phase = PHASE_RESURFACE
                self.max_divers_to_collect = 1  # Restart curriculum for Phase 2
                self.success_history.clear()
                print(f"  *** PHASE TRANSITION: Now in RESURFACE phase ***")
            else:
                # Phase 2 complete at max divers — training complete
                print(f"  *** ALL CURRICULUM COMPLETE ***")

    def get_max_reward(self):
        return self.max_reward

    def normalize_goal(self, goal):
        """Normalize goal to [0, 1] range using actual max values."""
        goal_max = torch.tensor([self.config.goal_max_divers, self.config.goal_max_resurface])
        return goal.float() / goal_max
