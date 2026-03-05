"""
Main training loop for DQN on the Seaquest environment.
"""

import argparse
import gymnasium as gym
import numpy as np
import os
import torch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from dqn import DQN, ReplayBuffer
from config import SeaQuestConfig
import ale_py

# Register the Atari environments with Gymnasium
gym.register_envs(ale_py)

# Environment Setup
env_name = "ALE/Seaquest-v5"  # RAM observation, no sticky actions [web:5]
env = gym.make(env_name, render_mode=None, obs_type="ram")

obs_size = env.observation_space.shape[0]  # 128 for RAM [web:16]
n_actions = env.action_space.n  # 18 for Seaquest [web:7]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
def train(n_iters=4000000, resume=False, use_ddqn=False, seed=0, output_dir="results"):
    """
    Training loop for DQN and Double DQN on the Seaquest environment.

    Args:
        n_iters:    Number of training iterations (environment steps) to run
        resume:     Whether to resume training from a checkpoint
        use_ddqn:   Whether to use Double DQN (if False, uses vanilla DQN)
        seed:       Random seed for reproducibility
        output_dir: Directory where results and checkpoints will be saved
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    obs = obs.astype(np.float32) / 255.0  # Normalize RAM [0,255] -> [0,1] [web:16]
    config = SeaQuestConfig()
    model = DQN(env=env, config=config)

    # Initialize the target action value as the model.
    target_model = DQN(env=env, config=config)
    target_model.load_state_dict(model.state_dict())
    replay_buffer = ReplayBuffer(state_dim=obs_size, capacity=config.replay_buffer_size, device=device)

    start_step = 0
    episode_rewards = []

    if resume:
        checkpoint = load_checkpoint(model, target_model)
        if checkpoint:
            start_step = checkpoint['step'] + 1
            episode_rewards = checkpoint['episode_rewards']
            print(f"Resumed from step {start_step}, {len(episode_rewards)} episodes, epsilon={model.epsilon:.4f}")
        else:
            print("No checkpoint found, starting fresh.")

    episode_reward = 0

    # Training loop for 4 million steps (can be adjusted as needed)
    for step in range(start_step, n_iters):

        action = model.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = next_obs.astype(np.float32) / 255.0
        done = terminated or truncated

        replay_buffer.push(obs, action, reward, next_obs, done)
        episode_reward += reward

        obs = next_obs

        # Train the model if the replay buffer has enough samples
        if replay_buffer.size >= 10000:
            # Train the model:x
            (state, next_state, action, rewards, terminal) = replay_buffer.sample(batch_size=config.batch_size)
            with torch.no_grad():
                if use_ddqn:
                    # Double DQN: online model selects action, target model evaluates
                    next_state_Q = model.forward(next_state)
                    target_next_state_Q = target_model.forward(next_state)
                    max_action = next_state_Q.max(dim=1).indices
                    max_action = max_action.reshape((max_action.shape[0], 1))
                    max_Q = torch.gather(target_next_state_Q, dim=1, index=max_action)
                else:
                    # Vanilla DQN: target model selects and evaluates
                    next_state_Q = target_model.forward(next_state)
                    max_Q = next_state_Q.max(dim=1).values
                    max_Q = max_Q.reshape((max_Q.shape[0], 1))

            # Get the obtained Q for the action:
            obtained_Q = model.forward(state)
            q_action = torch.gather(obtained_Q, dim=1, index=action)
            td_target = rewards + (config.gamma * max_Q * (1 - terminal.float()))

            # Compute the loss function:
            model.compute_loss(q_action.squeeze(), td_target.squeeze())

        if done:
            episode_rewards.append(episode_reward)
            print(f"Episode {len(episode_rewards)}, Reward: {episode_reward}, Steps: {step}")
            episode_reward = 0
            obs, _ = env.reset()
            obs = obs.astype(np.float32) / 255.0

        # Update the target model:
        # target_model = soft_update(model, target_model, tau=config.tau_weight)
        if step % config.target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())


        # Exponential decay of epsilon value:
        model.epsilon = max(1.0 - step / 1_000_000, 0.05)


    # Save scores, plot, and model checkpoint
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "scores.npy"), episode_rewards)
    export_plot(episode_rewards, "Score", "Seaquest DQN", os.path.join(output_dir, "scores.png"))
    save_checkpoint(model, target_model, step, episode_rewards)
    print(f"Saved scores.npy, scores.png, and checkpoint to {output_dir}/")

    return model


def soft_update(model, target_model, tau=0.005):
    """
    Performs a soft update of the target model's parameters towards the policy model's parameters.

    Args:
        model:          DQN model whose parameters will be used to update the target model
        target_model:   Target DQN model to be updated
        tau:            Interpolation parameter for the soft update (0 < tau <= 1)
    
    Returns:
        target_model:   Updated target model with parameters softly updated towards the policy model
    """

    # Perform polyak averaging instead of updating the model every C steps:
    policy_state_dict = model.state_dict()
    target_state_dict = target_model.state_dict()

    # Iterate over all the parameters in the model
    for k, v in policy_state_dict.items():
        target_state_dict[k] = tau * v + (1 - tau) * target_state_dict[k]

    target_model.load_state_dict(target_state_dict)

    return target_model


# Evaluation Template
def evaluate(model: DQN):  # Pass your trained DQN model
    """
    Evaluates the trained DQN model on the Seaquest environment.

    Args:
        model: Trained DQN model to be evaluated

    Returns:
        None (prints the average reward over 1000 evaluation episodes)
    """
    obs, _ = env.reset()
    obs = obs.astype(np.float32) / 255.0
    total_reward = 0

    # Trying to set the epsilon to a minimum value to avoid epsilon greedy action
    model.epsilon = 0.0001

    # Run 1000 evaluation episodes and compute the average reward
    for _ in range(1000):
        with torch.no_grad():
            action = model.select_action(obs)

        obs, reward, terminated, truncated, _ = env.step(action)
        obs = obs.astype(np.float32) / 255.0
        total_reward += reward

        if terminated or truncated:
            obs, _ = env.reset()
            obs = obs.astype(np.float32) / 255.0

    print(f"Average reward: {total_reward / 1000:.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=4000000, help="Number of training iterations (environment steps) to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--ddqn", action="store_true", help="Use Double DQN")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory where results and checkpoints will be saved")
    args = parser.parse_args()

    trained_model = train(n_iters=args.n_iters, resume=args.resume, use_ddqn=args.ddqn, seed=args.seed)
    evaluate(trained_model)
