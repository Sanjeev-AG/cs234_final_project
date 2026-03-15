"""
DQN with Hindsight Experience Replay (HER) using standard "future" strategy.

Both phases use the same HER relabeling:
- Sample k future states from [t+1, T]
- Use their achieved goal as the relabeled goal
- Skip trivial (already achieved or zero-diver) goals
- Set done=True on goal achievement to prevent Q-value explosion

With deposit-based diver tracking, achieved goals can be non-monotonic
(decrease on death/resurface penalty). HER naturally generates multi-resurface
goals like [4, 2] or [6, 3] as intermediate difficulty levels in Phase 2.

Sampling strategy: Mixed uniform + priority.
- 60% of batch: priority-sampled (focus on frontier goals and successes)
- 40% of batch: uniform-sampled (maintain earlier skills, prevent forgetting)
"""

import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, np2torch
from config import PHASE_COLLECTION, PHASE_RESURFACE


class ReplayBuffer(object):
    def __init__(self, state_dim, goal_dim, capacity, device, config, num_k=4):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.num_k = num_k
        self.config = config

        # Current curriculum state (updated externally when curriculum advances)
        self.frontier_divers = 1
        self.current_phase = PHASE_COLLECTION

        self.state = torch.zeros((capacity, state_dim), dtype=torch.float32).to(device)
        self.next_state = torch.zeros((capacity, state_dim), dtype=torch.float32).to(device)
        self.action = torch.zeros((capacity, 1), dtype=torch.int64).to(device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float).to(device)
        self.goal = torch.zeros((capacity, goal_dim), dtype=torch.float32).to(device)
        self.done = torch.zeros((capacity, 1), dtype=torch.uint8).to(device)
        self.priority = torch.ones((capacity,), dtype=torch.float32).to(device)

    def push(self, state, action, reward, next_state, done, goal):
        with torch.no_grad():
            self.state[self.ptr] = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            self.next_state[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
            self.action[self.ptr] = torch.as_tensor(action, dtype=torch.int64).to(self.device)
            self.reward[self.ptr] = torch.as_tensor(reward, dtype=torch.float32).to(self.device)
            self.done[self.ptr] = torch.as_tensor(done, dtype=torch.uint8).to(self.device)
            self.goal[self.ptr] = torch.as_tensor(goal, dtype=torch.float32).to(self.device)
            self.priority[self.ptr] = self._compute_priority(reward, goal)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _compute_priority(self, reward, goal):
        """
        Curriculum-relative priority instead of exponential.

        Old: 3^diver_count -> [1,0]=3, [3,0]=27, [6,0]=729 (243:1 ratio!)
        New: frontier=3, near=2, old=1 -> max 3:1 ratio (stable learning)
        """
        goal_divers = goal[0].item() if isinstance(goal, torch.Tensor) else goal[0]
        goal_resurface = goal[1].item() if isinstance(goal, torch.Tensor) else goal[1]
        reward_val = reward.item() if isinstance(reward, torch.Tensor) else reward

        # Curriculum-relative base weight
        if goal_divers >= self.frontier_divers:
            base = self.config.priority_frontier       # 3.0
        elif goal_divers >= self.frontier_divers - 1:
            base = self.config.priority_near_frontier   # 2.0
        else:
            base = self.config.priority_old             # 1.0

        # Success multiplier (positive reward transitions are rare and valuable)
        is_success = reward_val >= self.config.goal_reward * 0.9
        if is_success:
            base *= self.config.priority_success_mult   # 4x

        # Phase 2: resurface successes get extra boost (very rare, very valuable)
        if (self.current_phase == PHASE_RESURFACE
                and goal_resurface > 0 and is_success):
            base *= self.config.priority_resurface_mult  # 2x

        return base

    def sample(self, batch_size, use_priority=False):
        """
        Mixed sampling: 60% priority + 40% uniform.

        Why mixed? Pure priority causes catastrophic forgetting of easy goals.
        Pure uniform wastes capacity on already-mastered transitions.
        The mix balances frontier learning with skill maintenance.
        """
        if not use_priority:
            indices = np.random.randint(0, self.size, size=batch_size)
            return self._fetch_indices(indices)

        # Split batch: 60% priority, 40% uniform
        n_priority = int(batch_size * self.config.priority_batch_fraction)
        n_uniform = batch_size - n_priority

        # Priority portion
        candidate_count = min(n_priority * 16, self.size)
        candidate_indices = torch.randint(0, self.size, (candidate_count,), device=self.device)
        candidate_priorities = self.priority[candidate_indices]
        chosen = torch.multinomial(candidate_priorities, num_samples=n_priority, replacement=False)
        priority_indices = candidate_indices[chosen]

        # Uniform portion
        uniform_indices = torch.randint(0, self.size, (n_uniform,), device=self.device)

        # Combine
        all_indices = torch.cat([priority_indices, uniform_indices])
        return self._fetch_indices(all_indices)

    def _fetch_indices(self, indices):
        return (self.state[indices], self.next_state[indices], self.action[indices],
                self.reward[indices], self.done[indices], self.goal[indices])

    def update_curriculum_state(self, frontier_divers, current_phase):
        """Called when curriculum advances so priorities reflect the new frontier."""
        self.frontier_divers = frontier_divers
        self.current_phase = current_phase


def generate_her_transitions(replay_buffer, episode_states, episode_actions,
                             episode_next_states, episode_dones, achieved_goals,
                             current_phase, config, episode_deaths=None):
    """
    Generate HER transitions using the standard "future" strategy for both phases.

    For each transition t, sample k future indices from [t+1, T] and use their
    achieved goals as relabeled targets. This unified approach naturally produces
    multi-resurface goals in Phase 2 (e.g., [4, 2], [6, 3]) as intermediate
    difficulty levels.
    """
    T = len(episode_states)
    if T < 2:
        return

    her_k = config.her_k
    goal_reward = config.goal_reward

    for t in range(T):
        ag_current = achieved_goals[t]      # achieved at s_t
        ag_next = achieved_goals[t + 1]     # achieved at s_{t+1}

        # Sample k future indices from [t+1, T]
        future_range = range(t + 1, T + 1)
        k = min(her_k, len(future_range))
        if k == 0:
            continue
        future_indices = np.random.choice(list(future_range), size=k, replace=False)

        for fi in future_indices:
            relabeled_goal = achieved_goals[fi]

            # Skip if already achieved at current state (redundant transition)
            if torch.equal(ag_current, relabeled_goal):
                continue

            # Skip trivial goals [0, 0]
            if relabeled_goal[0] == 0:
                continue

            # Check if this transition achieves the relabeled goal
            is_achieved = torch.equal(ag_next, relabeled_goal)
            her_reward = goal_reward if is_achieved else 0.0
            her_done = episode_dones[t] or is_achieved  # Terminate on goal achievement

            if episode_deaths is not None and episode_deaths[t]:
                her_reward += config.death_penalty

            if current_phase == PHASE_COLLECTION:
                current_divers = ag_current[0].item()
                next_divers = ag_next[0].item()
                goal_divers = relabeled_goal[0].item()
                if next_divers > current_divers and current_divers < goal_divers:
                    rewarded_up_to = min(int(next_divers), int(goal_divers))
                    for i in range(int(current_divers) + 1, rewarded_up_to + 1):
                        her_reward += i * config.diver_milestone_bonus

            replay_buffer.push(episode_states[t], episode_actions[t], her_reward,
                               episode_next_states[t], her_done, relabeled_goal)


class DQN(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.lr = config.lr
        self.gamma = config.gamma

        observation_dim = self.env.observation_space.shape[-1] * config.stack_size
        if hasattr(env, 'num_goal_dimension'):
            observation_dim += env.num_goal_dimension
        if hasattr(env, 'num_extra_dimension'):
            observation_dim += env.num_extra_dimension

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = build_mlp(input_size=observation_dim, output_size=env.action_space.n,
                                 size=config.layer_size, n_layers=config.n_layers)
        self.network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.epsilon = config.epsilon

    def forward(self, obs, goal):
        if isinstance(obs, np.ndarray):
            obs = np2torch(obs)
        goal = goal.to(obs.device).float()
        obs = torch.cat((obs, goal), dim=-1)
        return self.network.forward(obs.float()).squeeze()

    def compute_loss(self, obtained_Q, target_Q):
        loss = torch.nn.functional.smooth_l1_loss(obtained_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()

    def select_action(self, in_state, goal=None):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            output = self.forward(in_state, goal)
            return torch.argmax(output).item()
