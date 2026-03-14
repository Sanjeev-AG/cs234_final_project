from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import numpy as np
import gymnasium as gym
from main import export_plot
import ale_py
import torch
import argparse

gym.register_envs(ale_py)


class EpisodeRewardCallback(BaseCallback):
    """Tracks and plots episode rewards during training."""
    def __init__(self, save_dir="results"):
        super().__init__()
        self.episode_count = 0
        self.episode_rewards = []
        self.save_dir = save_dir

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                reward = info["episode"]["r"]
                self.episode_rewards.append(reward)
                print(f"Episode {self.episode_count}: "
                      f"reward = {reward:.2f}, "
                      f"length = {info['episode']['l']}")

                # Save scores every 100 episodes
                if self.episode_count % 100 == 0:
                    self._save_scores()
        return True

    def _save_scores(self):
        scores = np.array(self.episode_rewards)
        np.save(f"{self.save_dir}/scores_dqn.npy", scores)
        export_plot(scores, "Reward", "DQN on Seaquest (RAM)", f"{self.save_dir}/scores_dqn.png")


# Environment Setup
env_name = "ALE/Seaquest-v5"
env = gym.make(env_name, render_mode=None, obs_type="ram")
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "results"


def train(resume=False, total_timesteps=10_000_000):
    if resume:
        print("Resuming from checkpoint...")
        model = DQN.load(f"{output_dir}/dqn_seaquest", env=env, device=device)
    else:
        model = DQN(
            "MlpPolicy",
            env,
            buffer_size=int(1e6),
            learning_starts=10000,
            batch_size=32,
            learning_rate=1e-4,
            gamma=0.99,
            target_update_interval=10000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            verbose=0,
            device=device,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=500_000,
        save_path=output_dir,
        name_prefix="dqn_seaquest",
    )
    reward_cb = EpisodeRewardCallback(save_dir=output_dir)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_cb, checkpoint_cb],
        reset_num_timesteps=not resume,
    )
    model.save(f"{output_dir}/dqn_seaquest_final")

    # Final save of scores
    reward_cb._save_scores()

    return model


def evaluate(model: DQN, num_episodes=10):
    obs, _ = env.reset()
    episode_reward = 0
    episode_rewards = []

    while len(episode_rewards) < num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            episode_rewards.append(episode_reward)
            print(f"Episode {len(episode_rewards)}: reward = {episode_reward:.2f}")
            episode_reward = 0
            obs, _ = env.reset()

    print(f"\nMean reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total training timesteps")
    args = parser.parse_args()

    trained_model = train(resume=args.resume, total_timesteps=args.timesteps)
    evaluate(trained_model)
