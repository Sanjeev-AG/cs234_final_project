"""
Main training loop for DQN with Hindsight Experience Replay (HER) on the Seaquest environment.
"""

import argparse
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import numpy as np
import os
import time
import torch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from dqn_her import DQN, ReplayBuffer
from config import SeaQuestConfig
import ale_py
from goal_wrapper import SeaQWrapper

# Register the Atari environments with Gymnasium
gym.register_envs(ale_py)

# Environment Setup
env_name = "ALE/Seaquest-v5"  # RAM observation, no sticky actions
env = gym.make(env_name, render_mode=None, obs_type="ram")

# env = FrameStackObservation(env=env, stack_size=4)
env = SeaQWrapper(env, SeaQuestConfig())
# obs_size = env.observation_space.shape[0] * env.observation_space.shape[1]
# state_offset = env.observation_space.shape[1] * (env.observation_space.shape[0] -1)
obs_size = env.observation_space.shape[0]
state_offset = 0
n_actions = env.action_space.n  # 18 for Seaquest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
output_dir = "results"

def export_plot(ys, ylabel, title, filename):
    """
    Create and save a plot of the given values.
    """
    plt.figure()
    plt.plot(range(len(ys)), ys)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def save_checkpoint(model, target_model, step, episode_rewards, dir):
    """
    Saves checkpoint of the training process for (D)DQN.

    Args:
        model:              DQN model to be saved
        target_model:       Target DQN model to be saved
        step:               Current training step to be saved
        episode_rewards:    List of episode rewards to be saved
        dir:                Directory where the checkpoint will be saved
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'step': step,
        'epsilon': model.epsilon,
        'episode_rewards': episode_rewards,
    }
    torch.save(checkpoint, os.path.join(dir, "checkpoint.pt"))

def load_checkpoint(model, target_model, dir):
    """
    Loads checkpoint of the training process for (D)DQN.

    Args:
        model:          DQN model to be loaded
        target_model:   Target DQN model to be loaded
        dir:            Directory where the checkpoint is located
    Returns:
        checkpoint:     Loaded checkpoint dictionary, or None if no checkpoint found
    """
    path = os.path.join(dir, "checkpoint.pt")
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_model.load_state_dict(checkpoint['target_model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.epsilon = checkpoint['epsilon']
    return checkpoint

# Training Loop Template
def train(n_iters=5000000, resume=False, seed=0, output_dir="results"):
    """
    Training loop for DDQN on the Seaquest environment.

    We use DDQN to avoid overestimation bias in the Q-values.

    Args:
        n_iters:    Number of training iterations (environment steps) to run
        resume:     Whether to resume training from a checkpoint
        seed:       Random seed for reproducibility
    """

    def reset_env(env, seed=None):
        if seed is not None:
            obs, _ = env.reset(seed=seed)
        else:
            obs, _ = env.reset()
        obs = np.concatenate((obs, [0]))
        obs = obs.astype(np.float32) / 255.0  # Normalize RAM [0,255] -> [0,1]
        return obs

    np.random.seed(seed)
    torch.manual_seed(seed)

    time_step = 0
    config = SeaQuestConfig()
    model = DQN(env=env, config=config)
    episode_step_idx = 0

    obtained_goals = []

    obs = reset_env(env, seed=seed)

    # Initialize the target action value as the model.
    target_model = DQN(env=env, config=config)
    target_model.load_state_dict(model.state_dict())
    replay_buffer = ReplayBuffer(state_dim=obs_size + env.num_extra_dimension, capacity=config.replay_buffer_size, device=device, goal_dim=env.num_goal_dimension, config=config)

    start_step = 0
    episode_rewards = []

    # Load checkpoint if resume is True
    if resume:
        checkpoint = load_checkpoint(model, target_model, output_dir)
        if checkpoint:
            start_step = checkpoint['step'] + 1
            episode_rewards = checkpoint['episode_rewards']
            print(f"Resumed from step {start_step}, {len(episode_rewards)} episodes, epsilon={model.epsilon:.4f}")
        else:
            print("No checkpoint found, starting fresh.")

    episode_reward = 0

    # Exponential decay of epsilon value:
    exploration_fraction = 1_000_000
    min_epsilon = 0.1

    success_idx = []
    backup_idx = []

    # Training loop for 5 million steps
    for step in range(start_step, n_iters):
        action = model.select_action(obs, goal=env.desired_goal/255) # Normalize the goals before passing to the model
        next_obs_raw, reward, terminated, truncated, reward_her, has_resurfaced = env.step(action)

        if reward_her > 45 or has_resurfaced:
            success_idx.append(episode_step_idx)
        elif env.achieved_goal[0] >= 1:
            backup_idx.append(episode_step_idx)

        episode_step_idx += 1

        next_obs = next_obs_raw.astype(np.float32) / 255.0

        # Extract Y vector from raw next observation before we scale it
        # RAM byte 97 is the submarine Y position
        done = terminated or truncated

        obtained_goals.append(env.get_achieved_goal())

        replay_buffer.push(obs, action, reward_her, next_obs, done, env.desired_goal)
        time_step += 1
        episode_reward += reward
        obs = next_obs

        # Train the model if the replay buffer has enough samples
        if replay_buffer.size >= 10000:
            (state, next_state, action_batch, rewards, terminal, goal) = replay_buffer.sample(
                batch_size=config.batch_size,
                use_priority=env.max_divers_to_collect > 1)

            goal = goal / 255
            # 1. Compute TD Targets WITHOUT gradients
            with torch.no_grad():
                # Double DQN logic
                next_state_q = model.forward(next_state, goal=goal)
                target_next_state_q = target_model.forward(next_state, goal=goal)
                max_action = next_state_q.max(dim=1).indices
                max_action = max_action.reshape((max_action.shape[0], 1))
                max_Q = torch.gather(target_next_state_q, dim=1, index=max_action)

                td_target = rewards + (config.gamma * max_Q * (1 - terminal.float()))

            # 2. Compute obtained Q values WITH gradients (outside the no_grad block)
            obtained_q = model.forward(state, goal=goal)
            q_action = torch.gather(obtained_q, dim=1, index=action_batch)

            # 3. Compute loss and backpropagate
            model.compute_loss(q_action.squeeze(), td_target.squeeze())

        if done:
            episode_rewards.append(episode_reward)

            print(f"Episode {len(episode_rewards)}, Reward: {episode_reward}, Steps: {step},"
                  f"Max reward in the episode: {env.get_max_reward()}, Desired goal: {env.desired_goal}")

            # Update the replay buffer with HER transitions:
            (state, next_state, action_batch, rewards_her, terminal, goal) = replay_buffer.fetch_last_N_samples(time_step)

            # Pass obtained_y_vectors to push_batch
            replay_buffer.push_batch(state, next_state, action_batch, rewards_her, terminal, obtained_goals, success_idx, backup_idx)

            # Reset the environment and episode trackers:
            time_step = 0
            episode_reward = 0
            episode_step_idx = 0
            obtained_goals = []

            obs= reset_env(env)
            success_idx = []
            backup_idx = []

        # Calculate the new epsilon
        decay_rate = (1.0 - min_epsilon) / exploration_fraction
        model.epsilon = max(1.0 - (step * decay_rate), min_epsilon)

        # Update the target model:
        if step % config.target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

    # Save scores, plot, and model checkpoint
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "scores.npy"), episode_rewards)
    export_plot(episode_rewards, "Score", "Seaquest DQN", os.path.join(output_dir, "scores.png"))
    save_checkpoint(model, target_model, step, episode_rewards, output_dir)
    print(f"Saved scores.npy, scores.png, and checkpoint to {output_dir}/")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=5000000, help="Number of training iterations (environment steps) to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory where results and checkpoints will be saved")
    args = parser.parse_args()

    t1 = time.time()
    trained_model = train(n_iters=args.n_iters, resume=args.resume, seed=args.seed, output_dir=args.output_dir)
    t2 = time.time()

    print(f"Device: {device}")
    print(f"Training completed in {int((t2 - t1) // 3600)}h:{int((t2 - t1) % 3600 // 60)}m:{int((t2 - t1) % 60)}s")
