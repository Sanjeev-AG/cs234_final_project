"""
Main training loop for DQN + HER on Seaquest with two-phase curriculum.

Key fixes from original:
1. Goal normalization: divide by actual max [6, 1], not 255
2. Two-phase curriculum: Phase 1 = collection only, Phase 2 = collection + resurface
3. Standard HER "future" strategy with done=True on goal achievement
4. Episode-based HER generation (cleaner than fetch-and-relabel)
5. Curriculum-aware epsilon: bumps on every curriculum advance, not just phase change
6. Mixed sampling: 60% priority + 40% uniform to prevent catastrophic forgetting
"""

import argparse
import gymnasium as gym
import numpy as np
import os
import time
import torch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from dqn_her import DQN, ReplayBuffer, generate_her_transitions
from config import SeaQuestConfig, PHASE_COLLECTION, PHASE_RESURFACE
import ale_py
from goal_wrapper import SeaQWrapper

gym.register_envs(ale_py)

env_name = "ALE/Seaquest-v5"
env = gym.make(env_name, render_mode=None, obs_type="ram")
env = SeaQWrapper(env, SeaQuestConfig())
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def export_plot(ys, ylabel, title, filename):
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def save_checkpoint(model, target_model, step, episode_rewards, env_wrapper,
                    epsilon_state, dir):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'step': step,
        'epsilon': model.epsilon,
        'episode_rewards': episode_rewards,
        'current_phase': env_wrapper.current_phase,
        'max_divers_to_collect': env_wrapper.max_divers_to_collect,
        'epsilon_state': epsilon_state,
    }
    torch.save(checkpoint, os.path.join(dir, "checkpoint.pt"))


def load_checkpoint(model, target_model, env_wrapper, dir):
    path = os.path.join(dir, "checkpoint.pt")
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_model.load_state_dict(checkpoint['target_model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.epsilon = checkpoint['epsilon']
    if 'current_phase' in checkpoint:
        env_wrapper.current_phase = checkpoint['current_phase']
        env_wrapper.max_divers_to_collect = checkpoint['max_divers_to_collect']
    return checkpoint


def normalize_goal(goal, config):
    """Normalize goal by actual max values, NOT 255."""
    goal_max = torch.tensor([config.goal_max_divers, config.goal_max_resurface])
    return goal.float() / goal_max


class EpsilonScheduler:
    """
    Curriculum-aware epsilon schedule.

    Instead of a single linear decay, this bumps epsilon on every curriculum
    advancement and decays it back down. This ensures the agent explores enough
    when facing new, harder goals.

    Schedule:
    - Phase 1 initial: 1.0 -> 0.1 over 1M steps
    - Phase 1 curriculum advance: bump to 0.2, decay to 0.1 over 200K steps
    - Phase 2 transition: bump to 0.3, decay to 0.05 over 500K steps
    - Phase 2 curriculum advance: bump to 0.15, decay to 0.05 over 200K steps
    """
    def __init__(self, config):
        self.config = config
        self.current_epsilon = config.phase1_epsilon_start
        self.target_epsilon = config.phase1_epsilon_min
        self.decay_rate = ((config.phase1_epsilon_start - config.phase1_epsilon_min)
                           / config.phase1_initial_decay_steps)
        self.current_phase = PHASE_COLLECTION

    def on_curriculum_advance(self, new_phase):
        """Called when curriculum advances (either within-phase or phase transition)."""
        cfg = self.config

        if new_phase != self.current_phase:
            # Phase transition: larger bump, lower floor
            self.current_epsilon = cfg.phase2_epsilon_reset
            self.target_epsilon = cfg.phase2_epsilon_min
            self.decay_rate = ((cfg.phase2_epsilon_reset - cfg.phase2_epsilon_min)
                               / cfg.phase2_epsilon_decay_steps)
            self.current_phase = new_phase
        else:
            # Within-phase advance: moderate bump
            if self.current_phase == PHASE_COLLECTION:
                bump = cfg.curriculum_epsilon_bump_p1
                floor = cfg.phase1_epsilon_min
            else:
                bump = cfg.curriculum_epsilon_bump_p2
                floor = cfg.phase2_epsilon_min

            self.current_epsilon = max(self.current_epsilon, bump)
            self.target_epsilon = floor
            self.decay_rate = (bump - floor) / cfg.curriculum_bump_decay_steps

    def step(self):
        """Decay epsilon by one step. Returns current epsilon."""
        self.current_epsilon = max(
            self.current_epsilon - self.decay_rate,
            self.target_epsilon
        )
        return self.current_epsilon

    def get_state(self):
        return {
            'current_epsilon': self.current_epsilon,
            'target_epsilon': self.target_epsilon,
            'decay_rate': self.decay_rate,
            'current_phase': self.current_phase,
        }

    def load_state(self, state):
        self.current_epsilon = state['current_epsilon']
        self.target_epsilon = state['target_epsilon']
        self.decay_rate = state['decay_rate']
        self.current_phase = state['current_phase']


def train(n_iters=10_000_000, resume=False, seed=0, output_dir="results", boost_epsilon=0.0):
    def reset_env(env, seed=None):
        if seed is not None:
            obs, _ = env.reset(seed=seed)
        else:
            obs, _ = env.reset()
        obs = np.concatenate((obs, [0]))
        obs = obs.astype(np.float32) / 255.0
        return obs

    np.random.seed(seed)
    torch.manual_seed(seed)

    config = SeaQuestConfig()
    model = DQN(env=env, config=config)
    target_model = DQN(env=env, config=config)
    target_model.load_state_dict(model.state_dict())
    target_model.network.eval()

    obs = reset_env(env, seed=seed)

    replay_buffer = ReplayBuffer(
        state_dim=obs_size + env.num_extra_dimension,
        capacity=config.replay_buffer_size,
        device=device,
        goal_dim=env.num_goal_dimension,
        config=config
    )

    eps_scheduler = EpsilonScheduler(config)

    start_step = 0
    episode_rewards = []
    episode_rewards_her = []

    if resume:
        checkpoint = load_checkpoint(model, target_model, env, output_dir)
        if checkpoint:
            start_step = checkpoint['step'] + 1
            episode_rewards = checkpoint['episode_rewards']
            if 'epsilon_state' in checkpoint and checkpoint['epsilon_state']:
                eps_scheduler.load_state(checkpoint['epsilon_state'])
            print(f"Resumed from step {start_step}, {len(episode_rewards)} episodes, "
                  f"epsilon={model.epsilon:.4f}, phase={env.current_phase}, "
                  f"max_divers={env.max_divers_to_collect}")
            if boost_epsilon:
                eps_scheduler.current_epsilon = boost_epsilon
                eps_scheduler.target_epsilon = config.phase1_epsilon_min
                eps_scheduler.decay_rate = ((boost_epsilon - config.phase1_epsilon_min)/ config.curriculum_bump_decay_steps)
                model.epsilon = boost_epsilon
        else:
            print("No checkpoint found, starting fresh.")

    episode_reward = 0
    episode_reward_her = 0

    # Track curriculum state for detecting advances
    prev_phase = env.current_phase
    prev_max_divers = env.max_divers_to_collect

    # Sync replay buffer with current curriculum state
    replay_buffer.update_curriculum_state(env.max_divers_to_collect, env.current_phase)

    # Episode buffer for HER
    ep_states = []
    ep_actions = []
    ep_next_states = []
    ep_dones = []
    ep_deaths = []
    achieved_goals = [env.get_achieved_goal()]

    for step in range(start_step, n_iters):
        goal_normalized = normalize_goal(env.desired_goal, config)
        action = model.select_action(obs, goal=goal_normalized)

        next_obs_raw, reward, terminated, truncated, reward_her, done = env.step(action)
        next_obs = next_obs_raw.astype(np.float32) / 255.0

        # Store in episode buffer
        ep_states.append(obs)
        ep_actions.append(action)
        ep_next_states.append(next_obs)
        ep_dones.append(done)
        ep_deaths.append(env._died_this_step)
        achieved_goals.append(env.get_achieved_goal())

        # Push original transition to replay buffer
        replay_buffer.push(obs, action, reward_her, next_obs, done, env.desired_goal)

        episode_reward += reward
        episode_reward_her += reward_her
        obs = next_obs

        # Train if enough samples
        if replay_buffer.size >= 10_000:
            use_priority = (env.max_divers_to_collect > 1
                            or env.current_phase == PHASE_RESURFACE)
            (state, next_state, action_batch, rewards, terminal, goal) = \
                replay_buffer.sample(batch_size=config.batch_size,
                                     use_priority=use_priority)

            goal_norm = normalize_goal(goal, config)

            with torch.no_grad():
                next_state_q = model.forward(next_state, goal=goal_norm)
                target_next_state_q = target_model.forward(next_state, goal=goal_norm)
                max_action = next_state_q.max(dim=1).indices.reshape(-1, 1)
                max_Q = torch.gather(target_next_state_q, dim=1, index=max_action)
                td_target = rewards + (config.gamma * max_Q * (1 - terminal.float()))

            obtained_q = model.forward(state, goal=goal_norm)
            q_action = torch.gather(obtained_q, dim=1, index=action_batch)
            model.compute_loss(q_action.squeeze(), td_target.squeeze())

        if done:
            episode_rewards.append(episode_reward)
            episode_rewards_her.append(episode_reward_her)

            phase_name = "COLLECT" if env.current_phase == PHASE_COLLECTION else "RESURFACE"
            print(f"Ep {len(episode_rewards)}, Phase={phase_name}, "
                  f"Reward={episode_reward:.0f}, HER_R={episode_reward_her:.0f}, "
                  f"Step={step}, Eps={model.epsilon:.3f}, "
                  f"Goal={env.desired_goal.tolist()}, "
                  f"MaxDivers={env.max_divers_to_collect}, "
                  f"PeakDivers={env.peak_divers}, "
                  f"DeathfromOxygen={env.deaths_from_oxygen}, "
                  f"DeathfromEnemy={env.deaths_from_enemy}")

            # Generate HER transitions from the completed episode
            generate_her_transitions(
                replay_buffer, ep_states, ep_actions, ep_next_states, ep_dones,
                achieved_goals, env.current_phase, config, ep_deaths
            )

            # Reset episode tracking
            episode_reward = 0
            episode_reward_her = 0
            ep_states = []
            ep_actions = []
            ep_next_states = []
            ep_dones = []
            ep_deaths = []
            obs = reset_env(env)
            achieved_goals = [env.get_achieved_goal()]

            # Detect curriculum advancement (phase change OR diver increase)
            if (env.current_phase != prev_phase
                    or env.max_divers_to_collect != prev_max_divers):
                eps_scheduler.on_curriculum_advance(env.current_phase)
                replay_buffer.update_curriculum_state(
                    env.max_divers_to_collect, env.current_phase)
                print(f"  *** Epsilon bumped to {eps_scheduler.current_epsilon:.3f} "
                      f"(target={eps_scheduler.target_epsilon:.3f}) ***")
                prev_phase = env.current_phase
                prev_max_divers = env.max_divers_to_collect

        # Update epsilon
        model.epsilon = eps_scheduler.step()

        # Update target model
        if step % config.target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())
            target_model.network.eval()

        # Periodic checkpoint
        if step > 0 and step % 2_000_000 == 0:
            os.makedirs(output_dir, exist_ok=True)
            save_checkpoint(model, target_model, step, episode_rewards, env,
                            eps_scheduler.get_state(), output_dir)
            np.save(os.path.join(output_dir, "scores.npy"), episode_rewards)
            np.save(os.path.join(output_dir, "scores_her.npy"), episode_rewards_her)
            print(f"Checkpoint saved at step {step}")

    # Final save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "scores.npy"), episode_rewards)
    np.save(os.path.join(output_dir, "scores_her.npy"), episode_rewards_her)
    export_plot(episode_rewards, "Score", "Seaquest DQN+HER",
                os.path.join(output_dir, "scores.png"))
    export_plot(episode_rewards_her, "HER Score", "Seaquest HER Reward",
                os.path.join(output_dir, "scores_her.png"))
    save_checkpoint(model, target_model, step, episode_rewards, env,
                    eps_scheduler.get_state(), output_dir)
    print(f"Saved results to {output_dir}/")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boost_epsilon", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    t1 = time.time()
    trained_model = train(n_iters=args.n_iters, resume=args.resume,
                          seed=args.seed, output_dir=args.output_dir, boost_epsilon=args.boost_epsilon)
    t2 = time.time()

    print(f"Device: {device}")
    print(f"Training completed in {int((t2 - t1) // 3600)}h:"
          f"{int((t2 - t1) % 3600 // 60)}m:{int((t2 - t1) % 60)}s")
