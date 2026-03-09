"""
Evaluation script for DQN+HER on Seaquest. Loads a checkpoint and renders the agent playing.
"""

import argparse
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import numpy as np
import os
import torch
import ale_py
from dqn_her import DQN
from config import SeaQuestConfig
from goal_wrapper import SeaQWrapper

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(checkpoint_path, n_episodes=100, seed=0):
    config = SeaQuestConfig()
    env = gym.make("ALE/Seaquest-v5", render_mode="human", obs_type="ram")
    env = FrameStackObservation(env=env, stack_size=config.stack_size)
    env = SeaQWrapper(env, config)

    model = DQN(env=env, config=config)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.epsilon = 0.01  # Near-greedy for evaluation

    step_count = checkpoint.get('step', 'unknown')
    n_train_episodes = len(checkpoint.get('episode_rewards', []))
    print(f"Loaded checkpoint: step={step_count}, training episodes={n_train_episodes}")

    # Set a high goal to push the agent to perform well
    env.desired_goal = torch.tensor([env.normalize_divers(6), env.get_oxygen_bucket(21)])

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = obs.astype(np.float32) / 255.0

        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = model.select_action(obs, goal=env.desired_goal)

            obs, reward, terminated, truncated, _ = env.step(action)
            obs = obs.astype(np.float32) / 255.0
            episode_reward += reward
            done = terminated or truncated

        print(f"Episode {ep + 1}/{n_episodes}, Reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/checkpoint.pt",
                        help="Path to the checkpoint file")
    parser.add_argument("--n_episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    evaluate(args.checkpoint, n_episodes=args.n_episodes, seed=args.seed)
